package uk.ac.soton.ecs.sag1g15;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;

import org.openimaj.util.array.ArrayUtils;
import org.openimaj.util.pair.IntDoublePair;
import org.springframework.cglib.core.ClassesKey;

import antlr.collections.impl.Vector;
import eu.larkc.csparql.parser.CSparqlParser.newVarFromExpression_return;

import org.apache.lucene.search.FieldCache.DoubleParser;
import org.bridj.cpp.std.vector;
import org.netlib.util.doubleW;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.experiment.evaluation.classification.BasicClassificationResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FloatFV;
import org.openimaj.feature.FloatFVComparison;
import org.openimaj.image.FImage;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.knn.DoubleNearestNeighboursExact;

public class KnnClassifier {
	
	private static final int k = 5;
	private LinkedHashMap<double[], String> featureVectors;
	private DoubleNearestNeighboursExact knn;

	public KnnClassifier() {
		
	}
	
//	public float[] extractFeatureVector(FImage image) {
//		float[][] imagePixels = getImagePixels(image);
//		float[] featurevector = new float[image.width * image.height];
//		int position = 0;
//		
//		for(int i = 0; i < imagePixels.length; i++) {
//			for(int j = 0; j < imagePixels[0].length; j++) {
//				featurevector[position] = imagePixels[i][j];
//				position++;
//			}
//		}
//		
//		return featurevector;
//	}
	
	public float[][] getImagePixels(FImage image) {
		return image.pixels;	
	}
	
	public void extract(GroupedDataset<String, ListDataset<FImage>, FImage> training) {
//		ArrayList<double[]> featureVectors = new ArrayList<double[]>();
//		ArrayList<String> classes = new ArrayList<String>();
//		
		this.featureVectors = new LinkedHashMap<double[], String>();
		
		VectorExtractor vectorExtractor = new VectorExtractor();
		//For each image in each class
		
		for(String group : training.getGroups()) {
			for(FImage image : training.get(group)) {
				DoubleFV featureVectorObject = vectorExtractor.extractFeature(image);
				featureVectorObject.normaliseFV();
				double[] featureVector = featureVectorObject.values;			
				featureVectors.put(featureVector, group);
			}
		}	
		
		double[][] vectors = featureVectors.keySet().toArray(new double[][] {});
			
		knn = new DoubleNearestNeighboursExact(vectors);
	}
	
	public BasicClassificationResult<String> classify(FImage image){
		
		VectorExtractor vectorExtractor = new VectorExtractor();	
		DoubleFV featureVectorObject = vectorExtractor.extractFeature(image);
		featureVectorObject.normaliseFV();
		
		double[] featureVector = featureVectorObject.values;
		
		List<IntDoublePair> neighbours = knn.searchKNN(featureVector, k);
		
		
	}
	
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
	
	
}
