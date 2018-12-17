package uk.ac.soton.ecs.sag1g15.run2;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FloatFV;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.LocalFeature;
import org.openimaj.feature.local.LocalFeatureImpl;
import org.openimaj.feature.local.SpatialLocation;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.image.pixel.sampling.RectangleSampler;
import org.openimaj.math.geometry.shape.Rectangle;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator.Mode;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.array.ArrayUtils;
import org.openimaj.util.pair.IntFloatPair;

import de.bwaldvogel.liblinear.SolverType;

/*
 * Linear Classifier using the LiblinearAnnotator to classify our images.
 * It uses a bag-of-visual-words feature based on a fixed size densely-sampled
 * pixel patch. Has an inner class called PatchFeatureExtractor which is used
 * to extract features from the image. Has a train, classifyImage, quantisy, 
 * generateClusters, extract, getFeatures and getResults methods.
 */

public class LinearClassifier {

	//Clusters to learn the vocabulary
    public final int clusters = 500;
    //Images for vocabulary
    public final int vocabulary = 10;

    //Step at which we sample the patch
	public final int step = 4;
	//Size of our patch
	public final int size = 8;

	//Liblinear Annotator used to classify images
	private LiblinearAnnotator<FImage, String> annotator;
	
	//GroupRandomSplitter to sample the data
	private GroupedRandomSplitter<String, FImage> randomSplitter;

	/*
	 * Function which takes the training set as a parameter. This function trains
	 * our training images. The GroupRandomSplitter takes 10 images for each class
	 * of images. It then uses a HardAssigner to assign features to identifiers. 
	 * A FeatureExractor is then created which takes the hardAssigner as a parameter
	 * and the Liblinear Annotator is instantiated with that extractor. The images are 
	 * then trained using the annotator.
	 */
	public void train(GroupedDataset<String, ListDataset<FImage>, FImage> trainingSet) {

		randomSplitter = new GroupedRandomSplitter<String, FImage>(trainingSet, vocabulary, 0, 0);
		HardAssigner<float[], float[], IntFloatPair> hardAssigner = quantisy(randomSplitter.getTrainingDataset());

		FeatureExtractor<DoubleFV, FImage> featureExtractor = new PatchFeatureExtractor(hardAssigner);

		annotator = new LiblinearAnnotator<FImage, String>(featureExtractor, Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
		
		System.out.println("Training started");
		annotator.train(trainingSet); 
		System.out.println("Training ended");
	}

	/*
	 * Function which uses the annotator to classify the image
	 */
	public ClassificationResult<String> classifyImage(FImage image) {
		return annotator.classify(image);
	}
	
	/*
	 * Function which quantises vectors. It takes a sample dataset as parameter and 
	 * creates a List of vectors. We iterate over the images from the dataset and extract
	 * the patches using our extract function. We then add the feature vectors to our vectors
	 * list. This list is then used with our generateClusters function to perform K-Means 
	 * clustering. The HardAssigner that assigned the features to identifiers is then returned.
	 */
	public HardAssigner<float[], float[], IntFloatPair> quantisy(Dataset<FImage> sample) {
		List<float[]> vectors = new ArrayList<float[]>();
  
		for (FImage image : sample) {
			List<LocalFeature<SpatialLocation, FloatFV>> sampleList = extract(image, step, size);

			for(LocalFeature<SpatialLocation, FloatFV> localFeature : sampleList){
				vectors.add(localFeature.getFeatureVector().values);
			}
		}
		System.out.println("Start clustering");
		FloatCentroidsResult result = generateClusters(vectors);
		System.out.println("End clustering");

		return result.defaultHardAssigner();
	}
	
	/*
	 * Function which performs our K-Means clustering based on our cluster variable
	 * declared globally. It takes a list of vectors as a parameter which are used
	 * to cluster the data.
	 */
	public FloatCentroidsResult generateClusters(List<float[]> vectors) {
		FloatKMeans km = FloatKMeans.createKDTreeEnsemble(clusters);
        float[][] data = vectors.toArray(new float[][]{});
        
        return km.cluster(data);
	}

	/*
	 * Function which extracts the patches from an image. Takes an FImage as a parameter,
	 * as well as a step and a size. For our patches, we have used 8x8 patches sampled every 4 pixels
	 * in the x and y directions. We create a rectangleSampler from our image using step and size and 
	 * loop over the rectangleSampler to extract the patches from the images. We then add the feature vectors
	 * and location to our patchList and return this list.
	 */
	public List<LocalFeature<SpatialLocation, FloatFV>> extract(FImage image, float step, float patch_size){
        List<LocalFeature<SpatialLocation, FloatFV>> patchList = new ArrayList<LocalFeature<SpatialLocation, FloatFV>>();

        RectangleSampler rectangleSampler = new RectangleSampler(image, step, step, patch_size, patch_size);

        for(Rectangle rectangle : rectangleSampler){
            FImage patch = image.extractROI(rectangle);
            LocalFeature<SpatialLocation, FloatFV> lf = getFeatures(patch, rectangle);

            patchList.add(lf);
        }

        return patchList;	
	}
	
	/*
	 * Function which returns location and featurevector of an image and rectangle.
	 * The location of the rectangle is where the location of the feature is.
	 */
	public LocalFeature<SpatialLocation, FloatFV> getFeatures(FImage image, Rectangle rectangle){
		float[] vector = ArrayUtils.reshape(image.pixels);
        FloatFV featureVector = new FloatFV(vector);

        SpatialLocation spatialLocation = new SpatialLocation(rectangle.x, rectangle.y);
        
        return new LocalFeatureImpl<SpatialLocation, FloatFV>(spatialLocation, featureVector);
	}
	
	/*
	 * Prints a detailed report of the classifier by displaying
	 * the accuracy and error rate.
	 */
	public void getResults(GroupedDataset<String, VFSListDataset<FImage>, FImage> training){

		randomSplitter = new GroupedRandomSplitter<String, FImage>(training, 15, 0, 15);
		
		ClassificationEvaluator<CMResult<String>, String, FImage> eval = 
		new ClassificationEvaluator<CMResult<String>, String, FImage>(
			annotator, randomSplitter.getTrainingDataset(), new CMAnalyser<FImage, String>(CMAnalyser.Strategy.SINGLE));
			
		Map<FImage, ClassificationResult<String>> guesses = eval.evaluate();
		CMResult<String> result = eval.analyse(guesses);

		System.out.println(result.getDetailReport());
	}
	
	/*
	 * PatchFeatureExtractor based on the hard assigner. It implements the 
	 * FeatureExtractor interface which has a Double feature vector and an FImage as 
	 * type parameters. The features of the image are extracted in respect to the HardAssigner
	 * using the extractFeature method.
	 */
	class PatchFeatureExtractor implements FeatureExtractor<DoubleFV, FImage> {
		HardAssigner<float[], float[], IntFloatPair> assigner;

		public PatchFeatureExtractor(HardAssigner<float[], float[], IntFloatPair> assigner) {
			this.assigner = assigner;
		}

		public DoubleFV extractFeature(FImage image) {
			BagOfVisualWords<float[]> bovw = new BagOfVisualWords<float[]>(assigner);
			BlockSpatialAggregator<float[], SparseIntFV> spatial = new BlockSpatialAggregator<float[], SparseIntFV>(bovw, 2, 2);
			return spatial.aggregate(extract(image, step, size), image.getBounds()).normaliseFV();
		}
	}

}
