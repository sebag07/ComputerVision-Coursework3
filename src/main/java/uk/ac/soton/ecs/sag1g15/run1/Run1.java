package uk.ac.soton.ecs.sag1g15.run1;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import org.apache.commons.vfs2.FileSystemException;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.util.DatasetAdaptors;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;

import uk.ac.soton.ecs.sag1g15.KnnClassifier;


/**
 * OpenIMAJ Hello world!
 *
 */
public class Run1 {

	public static void main(String[] args) throws FileSystemException {
		writeFile();
	}
	
	public static void writeFile() throws FileSystemException {
		VFSListDataset<FImage> testing = new VFSListDataset<>("C:\\Users\\tzica\\Coursework3\\testing", ImageUtilities.FIMAGE_READER);
		
		GroupedDataset<String, VFSListDataset<FImage>, FImage> training = new VFSGroupDataset<>("C:\\Users\\tzica\\Coursework3\\training", ImageUtilities.FIMAGE_READER);
		
		System.out.println(testing.size());
		System.out.println(training.size());
		
		List<FImage> testingImages = DatasetAdaptors.asList(testing);
		List<FImage> trainingImages = DatasetAdaptors.asList(training);
		
		KnnClassifier classifier = new KnnClassifier();
		
		LinkedHashMap<double[], String> featureVectors = classifier.train(training);
		
		classifier.train(training);
		
		BufferedWriter writer = null;
		try {
			FileWriter fw = new FileWriter("run1.txt");
			for(int i = 0; i < testing.size(); i++) {
				FImage newimage = testingImages.get(i);
				System.out.println(i + ".jpg " + classifier.classify(newimage, 7, featureVectors));
				fw.write(i + ".jpg " + classifier.classify(newimage, 7, featureVectors) + "\n");	
			}
			fw.close();
		} catch(Exception e) {
			System.out.println(e);
		}
	}
}
