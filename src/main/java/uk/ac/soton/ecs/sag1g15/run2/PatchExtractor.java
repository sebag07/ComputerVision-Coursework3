package uk.ac.soton.ecs.sag1g15.run2;

import java.util.ArrayList;
import java.util.List;

import org.openimaj.feature.FloatFV;
import org.openimaj.image.FImage;

public class PatchExtractor {

	
	protected int size;
	protected int sample;
	
	public PatchExtractor(int size, int sample) {
		this.size = size;
		this.sample = sample;
	}
	
	public ArrayList<FImage> getPatches(FImage image){
		ArrayList<FImage> patches = new ArrayList<FImage>();
		
		for(int i = 0; i < image.width - size; i += sample) {		
			for(int j = 0; j < image.height - size; j += sample) {		
				FImage patch = image.extractROI(i, j, size, size);
				
				patches.add(patch.normalise());
			}
		}
		
		return patches;
	} 

	
	public FloatFV extractVector(FImage patch) {
		int width = patch.width;
		int height = patch.height;
		float[] vector = new float[width * height];
		
		for(int i = 0; i < height; i++) {
			for( int j = 0; j < width; j++) {
				vector[i*width+j] = patch.pixels[i][j];
			}
		}
		
		return new FloatFV(vector);
	}
	
	public int getSize() {
		return this.size;
	}
	
	public int getSample() {
		return this.sample;
	}
	
}
