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


public class LinearClassifier {

    public final int CLUSTERS = 500;
    public final int vocabulary = 10;

	public final int step = 4;
	public final int size = 8;

	private LiblinearAnnotator<FImage, String> annotator;
	private GroupedRandomSplitter<String, FImage> randomSplitter;

	public void train(GroupedDataset<String, ListDataset<FImage>, FImage> trainingSet) {

		randomSplitter = new GroupedRandomSplitter<String, FImage>(trainingSet, vocabulary, 0, 0);
		HardAssigner<float[], float[], IntFloatPair> hardAssigner = quantisy(randomSplitter.getTrainingDataset());

		FeatureExtractor<DoubleFV, FImage> featureExtractor = new PatchFeatureExtractor(hardAssigner);

		annotator = new LiblinearAnnotator<FImage, String>(featureExtractor, Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
		
		System.out.println("Training started");
		annotator.train(trainingSet); 
		System.out.println("Training ended");
	}


	public ClassificationResult<String> classifyImage(FImage image) {
		return annotator.classify(image);
	}
	
	
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
	
	public FloatCentroidsResult generateClusters(List<float[]> vectors) {
		FloatKMeans km = FloatKMeans.createKDTreeEnsemble(CLUSTERS);
        float[][] data = vectors.toArray(new float[][]{});
        
        return km.cluster(data);
	}

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
	
	public LocalFeature<SpatialLocation, FloatFV> getFeatures(FImage image, Rectangle rectangle){
		float[] vector = ArrayUtils.reshape(image.pixels);
        FloatFV featureVector = new FloatFV(vector);

        SpatialLocation spatialLocation = new SpatialLocation(rectangle.x, rectangle.y);
        
        return new LocalFeatureImpl<SpatialLocation, FloatFV>(spatialLocation, featureVector);
	}
	
	public void getResults(GroupedDataset<String, VFSListDataset<FImage>, FImage> training){

		randomSplitter = new GroupedRandomSplitter<String, FImage>(training, 15, 0, 15);
		
		ClassificationEvaluator<CMResult<String>, String, FImage> eval = 
		new ClassificationEvaluator<CMResult<String>, String, FImage>(
			annotator, randomSplitter.getTrainingDataset(), new CMAnalyser<FImage, String>(CMAnalyser.Strategy.SINGLE));
			
		Map<FImage, ClassificationResult<String>> guesses = eval.evaluate();
		CMResult<String> result = eval.analyse(guesses);

		System.out.println(result.getDetailReport());
	}
   
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
