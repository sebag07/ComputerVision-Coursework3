package uk.ac.soton.ecs.sag1g15.run1;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

import org.apache.commons.vfs2.FileSystemException;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.util.DatasetAdaptors;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;

import info.bliki.wiki.filter.StringPair;


/**
 * OpenIMAJ Hello world!
 *
 */
public class Run1 {
	
	public static void main(String[] args) throws FileSystemException {
		File testingFile = new File("./testing/");
		VFSListDataset<FImage> testing = new VFSListDataset<>(testingFile.getAbsolutePath(), ImageUtilities.FIMAGE_READER);
		
		File trainingFile = new File("./training/");
		GroupedDataset<String, VFSListDataset<FImage>, FImage> training = new VFSGroupDataset<>(trainingFile.getAbsolutePath(), ImageUtilities.FIMAGE_READER);
		
		KnnClassifier classifier = new KnnClassifier(15);
//		
//		LinkedHashMap<double[], String> featureVectors = classifier.train(training);
//		
		classifier.train(training);
		
		TreeMap<Integer, String> predictedclasses = new TreeMap<Integer, String>();
		
		printPredictedClasses(testing, classifier, predictedclasses);
		
		writeToFile(predictedclasses);
		
		classifier.getResults(training);
	}
	
	public static void writeToFile(TreeMap<Integer, String> predictedclasses) {
		try {
			FileWriter fw = new FileWriter("run1.txt");
			System.out.println("Writing output to run1.txt");
			for(Map.Entry<Integer, String> classes : predictedclasses.entrySet()) {
				fw.write(classes.getKey() + ".jpg " + classes.getValue() + "\n");
			}
			fw.close(); 
			System.out.println("File is ready");
		} catch(Exception e) {
			System.out.println(e);
		}
	}
	
	public static void printPredictedClasses(VFSListDataset<FImage> testing, KnnClassifier classifier, TreeMap<Integer, String> predictedclasses) {
		System.out.println("Classifying images");
		
		for(int i = 0; i < testing.size(); i++) {
			FImage testImage = testing.get(i);
			ClassificationResult<String> predicted = classifier.classifyImage(testImage);
			Set<String> predictedClass = predicted.getPredictedClasses();
			String first = predictedClass.stream().findFirst().get();
			String file = testing.getID(i);
			String[] parts = file.split(".jpg");
			String fileNumber = parts[0];
			int number = Integer.parseInt(fileNumber);
			System.out.println(file + " " + first);
			predictedclasses.put(number, first);
		}
		
		System.out.println("Classifying done");
	}
	
	
}
