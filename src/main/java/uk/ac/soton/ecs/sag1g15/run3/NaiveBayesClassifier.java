package uk.ac.soton.ecs.sag1g15.run3;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.sampling.GroupSampler;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.image.feature.local.aggregate.PyramidSpatialAggregator;
import org.openimaj.ml.annotation.bayes.NaiveBayesAnnotator;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.ml.kernel.HomogeneousKernelMap;
import org.openimaj.ml.kernel.HomogeneousKernelMap.KernelType;
import org.openimaj.ml.kernel.HomogeneousKernelMap.WindowType;
import org.openimaj.util.pair.IntFloatPair;


/*
 * NaiveBayesClassifier using the NaiveBayesAnnotator to classify our images.
 * It uses a pyramid histogram of words together with Dense SIFT spatial pooling
 * to assign identifiers to our images. The identifier is then assigned by training
 * a vector quantiser using k-means. Has an inner class PHOWExtractor that implements
 * the FeatureVector interface. Has a Double feature vector and a FImage as type parameters.
 * The inner class uses a PyramidSpatialAggregator together with a BagOfVisualWords to
 * compute the histograms across the image. It then uses the corresponding hardAssigner
 * to assign Dense SIFT features to a visual word. The resultant histograms are appended
 * together and returned as a normalised vector. The NaiveBayesClassifier class has a
 * train, classifyImage, trainQuantiser and getResults methods.
 */

public class NaiveBayesClassifier {
	
	//Bayes classifier
	private NaiveBayesAnnotator<FImage, String> annotator;

	private GroupedDataset<String, ListDataset<FImage>, FImage> data;

	private GroupedRandomSplitter<String, FImage> splits;
	//Number of clusters used for K-Means
	private final int clusters = 300;
	
	/*
	 * Our train functions which takes our training set as a parameter. We instantiate
	 * our Dense SIFT pyramid and assign the SIFT features to a visual word using the HardAssigner.
	 * We then create a HomogenousKernelMap of type Chi.2 which we use with our feature extractor.
	 * We then instantiate our annotator and train our training set.
	 */
	public void train(GroupedDataset<String, VFSListDataset<FImage>, FImage> training) throws FileSystemException {
		
		
		data = GroupSampler.sample(training, training.size(), false);

		splits = new GroupedRandomSplitter<String, FImage>(data, 15, 0, 0);
		
		DenseSIFT dsift = new DenseSIFT(3, 7);
		PyramidDenseSIFT<FImage> pdsift = new PyramidDenseSIFT<FImage>(dsift, 6f, 7);
		
		HardAssigner<byte[], float[], IntFloatPair> assigner = trainQuantiser(splits.getTrainingDataset(), pdsift);
	
		HomogeneousKernelMap hKernelMap = new HomogeneousKernelMap(KernelType.Chi2, WindowType.Rectangular);
		
		FeatureExtractor<DoubleFV, FImage> extractor = hKernelMap.createWrappedExtractor(new PHOWExtractor(pdsift, assigner));
		
		annotator = new NaiveBayesAnnotator<FImage, String>(
	            extractor, NaiveBayesAnnotator.Mode.MAXIMUM_LIKELIHOOD);
		System.out.println("Start training");
		annotator.train(splits.getTrainingDataset());
		System.out.println("Training done");
		
	}
	
	/*
	 * Function which uses the annotator to classify an image.
	 */
	public ClassificationResult<String> classifyImage(FImage image){
		return annotator.classify(image);
	}
	
	/*
	 * Prints a detailed report of the classifier by displaying
	 * the accuracy and error rate.
	 */
	public void getResults(GroupedDataset<String, VFSListDataset<FImage>, FImage> training){

		splits = new GroupedRandomSplitter<String, FImage>(data, 15, 0, 15);
		
		ClassificationEvaluator<CMResult<String>, String, FImage> eval = 
		new ClassificationEvaluator<CMResult<String>, String, FImage>(
			annotator, splits.getTrainingDataset(), new CMAnalyser<FImage, String>(CMAnalyser.Strategy.SINGLE));
			
		Map<FImage, ClassificationResult<String>> guesses = eval.evaluate();
		CMResult<String> result = eval.analyse(guesses);

		System.out.println(result.getDetailReport());
	}
	
	/*
	 * Function which returns a HardAssigner that is used to assign SIFT features to identifiers
	 * It takes our training set as a parameter as well as our PyramidDenseSIFT object.
	 * We create a list of sift features from the training set. For each image, we loop and get the sift
	 * features. The sift features are reduced to be 10000. The K-Means classifier has been created
	 * using 300 clusters. The number of clusters can be increased and the classifier will receive a higher
	 * accuracy, but the time to run the program is significantly higher. We then generate clusters from the
	 * sift features.
	 */
	public HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(GroupedDataset<String, ListDataset<FImage>, FImage> groupedDataset, PyramidDenseSIFT<FImage> pdsift){
		
		 List<LocalFeatureList<ByteDSIFTKeypoint>> allkeys = new ArrayList<LocalFeatureList<ByteDSIFTKeypoint>>();
		
		 System.out.println("Sift train has started");
		 
		 for(FImage rec : groupedDataset) {
			 FImage image = rec.getImage();
			 
			 pdsift.analyseImage(image);
			 allkeys.add(pdsift.getByteKeypoints(0.005f));
		 }
		 
		 System.out.println("Sift train has ended");
		 
		 if(allkeys.size() > 10000) {
			 allkeys = allkeys.subList(0, 10000);
		 }
		 //KMeans classifier with 300 visual words
		 ByteKMeans kMeans = ByteKMeans.createKDTreeEnsemble(clusters);
		 DataSource<byte[]> datasource = new LocalFeatureListDataSource<ByteDSIFTKeypoint, byte[]>(allkeys);
		 System.out.println("Clustering has started");
		 //Clusters from sift features
		 ByteCentroidsResult result = kMeans.cluster(datasource);
		 System.out.println("Clustering done");
		 
		 return result.defaultHardAssigner();
	}
	
	/*
	 * Inner class that extracts a bag of visual words feature vector
	 * In the extractFeature function we get the sift features of the input image,
	 * compute the Bag of visual words histogram representation and then use the
	 * PyramidSpatialAggregator with the bag of visual words to return the
	 * normalised feature vector.
	 */
	class PHOWExtractor implements FeatureExtractor<DoubleFV, FImage> {

		PyramidDenseSIFT<FImage> pdsift;
		HardAssigner<byte[], float[], IntFloatPair> assigner;
		
		public PHOWExtractor(PyramidDenseSIFT<FImage> pdsift,
		HardAssigner<byte[], float[], IntFloatPair> assigner) {
			this.pdsift = pdsift;
			this.assigner = assigner;
		}
		
		@Override
		public DoubleFV extractFeature(FImage object) {
			FImage image = object.getImage();
			pdsift.analyseImage(image);
			
			BagOfVisualWords<byte[]> bagOfVisualWords = new BagOfVisualWords<byte[]>(assigner);
		
			PyramidSpatialAggregator<byte[], SparseIntFV> spatial = new PyramidSpatialAggregator<byte[], SparseIntFV>(bagOfVisualWords, 2, 4);
		
			return spatial.aggregate(pdsift.getByteKeypoints(0.015f), image.getBounds()).normaliseFV();
			
		}
		
	}
	
}