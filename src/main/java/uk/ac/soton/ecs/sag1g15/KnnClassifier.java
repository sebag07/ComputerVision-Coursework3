package uk.ac.soton.ecs.sag1g15;

import org.openimaj.image.FImage;

public class KnnClassifier {
	
	

	public KnnClassifier() {
		
	}
	
	public float[] extractFeatureVector(FImage image) {
		float[][] imagePixels = getImagePixels(image);
		float[] featurevector = new float[image.width * image.height];
		int position = 0;
		
		for(int i = 0; i < imagePixels.length; i++) {
			for(int j = 0; j < imagePixels[0].length; j++) {
				featurevector[position] = imagePixels[i][j];
				position++;
			}
		}
		
		return featurevector;
	}
	
	public float[][] getImagePixels(FImage image) {
		return image.pixels;
	}
	
	public int calculateDistance(float[] testingvector, float[] trainingvector) {
		int sum = 0;
		
		for (int i = 0; i < trainingvector.length; i++) {
			sum += (testingvector[i] - trainingvector[i]) * (testingvector[i] - trainingvector[i]);
		}
		
		return sum;
	}
	
	
}
