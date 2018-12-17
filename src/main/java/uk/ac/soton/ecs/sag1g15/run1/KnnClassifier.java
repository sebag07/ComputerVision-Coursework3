package uk.ac.soton.ecs.sag1g15.run1;

import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.PriorityQueue;

import org.openimaj.util.array.ArrayUtils;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.DoubleFVComparison;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.image.FImage;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.ml.annotation.basic.KNNAnnotator;


/*
 * K-nearest-neighbour classifier using the "tiny image" feature.
 * Has a train, classifyImage and getResults methods, as well as 
 * an inner class called VectorExtractor which extracts the
 * tiny image features.
 */

public class KnnClassifier {

	private final int k;
	private KNNAnnotator<FImage, String, DoubleFV> annotator;
	
	/*
	 * Constructor which takes the k value as a parameter
	 */
	public KnnClassifier(int k) {
		this.k = k;
	}
	
	/*
	 * Method which returns the pixel values of the image
	 * in a float[][]
	 */
	public float[][] getImagePixels(FImage image) {
		return image.pixels;	
	}
	
	/*
	 * Train function which takes our training set as a parameter.
	 * It creates a VectorExtractor, a DoubleFV comparator 
	 * to compare Euclidean distances and creates our KNNAnnotator 
	 * which is used to train our training set.
	 */
	public void train(GroupedDataset<String, VFSListDataset<FImage>, FImage> training) {
		VectorExtractor vectorExtractor = new VectorExtractor();
		DoubleFVComparison comparator = DoubleFVComparison.EUCLIDEAN;
		annotator = KNNAnnotator.create(vectorExtractor, comparator, k);
		
		annotator.train(training);
	}
	
	/*
	 * Classifies the given image using the KNNAnnotator
	 */
	public ClassificationResult<String> classifyImage(FImage image){
		return annotator.classify(image);
	}
	
	/*
	 * Prints a detailed report of the classifier by displaying
	 * the accuracy and error rate.
	 */
	public void getResults(GroupedDataset<String, VFSListDataset<FImage>, FImage> training){

		GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<String, FImage>(training, 15, 0, 15);
		
		ClassificationEvaluator<CMResult<String>, String, FImage> eval = 
		new ClassificationEvaluator<CMResult<String>, String, FImage>(
			annotator, splits.getTrainingDataset(), new CMAnalyser<FImage, String>(CMAnalyser.Strategy.SINGLE));
			
		Map<FImage, ClassificationResult<String>> guesses = eval.evaluate();
		CMResult<String> result = eval.analyse(guesses);

		System.out.println(result.getDetailReport());
	}
	
	/*
	 * Inner class used to extract the tiny Image feature from an image.
	 * It has a static final int square_size which is the size of the tiny
	 * image, in this case it is 16x16. In the extractFeature method
	 * we square the image by extracting the center giving the smallest dimension,
	 * we then resize the image to 16x16 and then return a Double feature vector as
	 * a 1D vector
	 */
	class VectorExtractor implements FeatureExtractor<DoubleFV, FImage>{

		//Size of tiny image
		private static final int square_size = 16;
	
		public DoubleFV extractFeature(FImage image) {
			int imageSize = Math.min(image.width, image.height);
			
			FImage center = image.extractCenter(imageSize, imageSize);
			
			FImage small = center.process(new ResizeProcessor(square_size, square_size));
			
			return new DoubleFV(ArrayUtils.reshape(ArrayUtils.convertToDouble(small.pixels)));
		}
		
	}
	
}