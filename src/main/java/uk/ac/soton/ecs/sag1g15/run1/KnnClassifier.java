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

public class KnnClassifier {

	private final int k;
	private KNNAnnotator<FImage, String, DoubleFV> annotator;
	
	public KnnClassifier(int k) {
		this.k = k;
	}
	
	public float[][] getImagePixels(FImage image) {
		return image.pixels;	
	}
	
	public void train(GroupedDataset<String, VFSListDataset<FImage>, FImage> training) {
		VectorExtractor vectorExtractor = new VectorExtractor();
		DoubleFVComparison comparator = DoubleFVComparison.EUCLIDEAN;
		annotator = KNNAnnotator.create(vectorExtractor, comparator, k);
		
		annotator.train(training);
	}
	
	public ClassificationResult<String> classifyImage(FImage image){
		return annotator.classify(image);
	}
	
	public void getResults(GroupedDataset<String, VFSListDataset<FImage>, FImage> training){

		GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<String, FImage>(training, 15, 0, 15);
		
		ClassificationEvaluator<CMResult<String>, String, FImage> eval = 
		new ClassificationEvaluator<CMResult<String>, String, FImage>(
			annotator, splits.getTrainingDataset(), new CMAnalyser<FImage, String>(CMAnalyser.Strategy.SINGLE));
			
		Map<FImage, ClassificationResult<String>> guesses = eval.evaluate();
		CMResult<String> result = eval.analyse(guesses);

		System.out.println(result.getDetailReport());
	}
	
//	public LinkedHashMap<double[], String> train(GroupedDataset<String, VFSListDataset<FImage>, FImage> training) {
//
//		LinkedHashMap<double[], String> featureVectors = new LinkedHashMap<double[], String>();
//		
//		VectorExtractor vectorExtractor = new VectorExtractor();
//		//For each image in each class
//		
//		for(String group : training.getGroups()) {
//			for(FImage image : training.get(group)) {
//				DoubleFV featureVectorObject = vectorExtractor.extractFeature(image);
//				featureVectorObject.normaliseFV();
//				double[] featureVector = featureVectorObject.values;			
//				featureVectors.put(featureVector, group);
//			}
//		}	
//		
//		return featureVectors;
//	}
//	
//	
//	
//	public String classify(FImage image, int kValue, LinkedHashMap<double[], String> featureVectors) {
//		
//		VectorExtractor vectorExtractor = new VectorExtractor();
//		//For each image in each class
//		
//		DoubleFV featureVectorObject = vectorExtractor.extractFeature(image);	
//		featureVectorObject.normaliseFV();
//		double[] featureVector = featureVectorObject.values;
//		
//		PriorityQueue<double[]> q = new PriorityQueue<double[]>(1, new VectorComparison(featureVector));
//		
//		for(double[] vector : featureVectors.keySet()) {
//			q.add(vector);
//		}
//		
//		HashMap<String, Integer> classCount = new HashMap<String, Integer>();
//		
//		for(int i = 0; i < kValue; i++) {
//			double[] neighbour = q.poll();
//			String neighbourClass = featureVectors.get(neighbour);
//			
//			if(classCount.containsKey(neighbourClass)) {
//				classCount.put(neighbourClass, classCount.get(neighbourClass) + 1);
//			} else {
//				classCount.put(neighbourClass, new Integer(1));
//			}
//		}
//		
//		String targetClass = null;
//		Integer n = 0;
//		
//		for(String string : classCount.keySet()) {
//			Integer stringNumber = classCount.get(string);
//			if(stringNumber > n) {
//				targetClass = string;
//				n= stringNumber;
//			}
//		}
//		
//		return targetClass;
//	}
//	
//	public double getDistance(double[] v1, double[] v2) {		
//		double diff_square_sum = 0;
//		for(int i = 0; i < v1.length; i++) {
//			diff_square_sum += (v1[i] - v2[i]) * (v1[i] - v2[i]);
//		}
//		
//		return Math.sqrt(diff_square_sum);
//	}
	
	/*
	 * Extracts TinyImage feature vector from an image
	 */
	
	class VectorExtractor implements FeatureExtractor<DoubleFV, FImage>{

		//Size of tiny image
		private static final int square_size = 16;
	
		public DoubleFV extractFeature(FImage image) {
			//Sets the size of the image given the smallest dimension
			//of the image. It is the biggest the square image can be
			int imageSize = Math.min(image.width, image.height);
			
			//Extracts the square from the center
			FImage center = image.extractCenter(imageSize, imageSize);
			
			//Resizes the image to a tiny image 16x16
			FImage small = center.process(new ResizeProcessor(square_size, square_size));
			
			//Returns the feature vector (2D array to 1D vector)
			return new DoubleFV(ArrayUtils.reshape(ArrayUtils.convertToDouble(small.pixels)));
		}
		
	}
	
//	class VectorComparison implements Comparator<double[]>{
//
//		double[] vector;
//		
//		public VectorComparison(double[] vector) {
//			this.vector = vector;
//		}
//		
//		@Override
//		public int compare(double[] o1, double[] o2) {
//			DoubleFVComparison comparison = DoubleFVComparison.EUCLIDEAN;
//			Double dist0 = comparison.compare(o1, vector);
//			Double dist1 = comparison.compare(o2, vector);
//			
//			return dist0.compareTo(dist1);
//		}
//		
//	}
	
}