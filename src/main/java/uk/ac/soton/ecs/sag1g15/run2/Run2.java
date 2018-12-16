package uk.ac.soton.ecs.sag1g15.run2;

import java.io.File;
import java.io.FileWriter;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.sampling.GroupSampler;
import org.openimaj.experiment.dataset.util.DatasetAdaptors;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;

public class Run2 {

	public static void main(String[] args) throws FileSystemException {
		LinearClassifier linearClassifier = new LinearClassifier();
		
		File testingFile = new File("./testing/");
		VFSListDataset<FImage> testing = new VFSListDataset<>(testingFile.getAbsolutePath(), ImageUtilities.FIMAGE_READER);

		File trainingFile = new File("./training/");
		GroupedDataset<String, VFSListDataset<FImage>, FImage> training = new VFSGroupDataset<>(trainingFile.getAbsolutePath(), ImageUtilities.FIMAGE_READER);
		
		GroupedDataset<String, ListDataset<FImage>, FImage> data = GroupSampler.sample(training, training.size(), false);
		
		linearClassifier.train(data);
		
		TreeMap<Integer, String> predictedclasses = new TreeMap<Integer, String>();
		
		printPredictedResults(testing, linearClassifier, predictedclasses);
		
		writeToFile(predictedclasses);
		
		linearClassifier.getResults(training);
		
	}
	
	public static void printPredictedResults(VFSListDataset<FImage> testing, LinearClassifier linearClassifier, TreeMap<Integer, String> predictedclasses) {
		System.out.println("Classifying images");
		
		for(int i = 0; i < testing.size(); i++) {
			ClassificationResult<String> predicted = linearClassifier.classifyImage(testing.get(i));
			String pred = predicted.getPredictedClasses().toString();
			String file = testing.getID(i);
			String[] parts = file.split(".jpg");
			String fileNumber = parts[0];
			int number = Integer.parseInt(fileNumber);
			pred = pred.replaceAll("\\[", "").replaceAll("\\]", "");
			System.out.println(file + " " + pred);
			predictedclasses.put(number, pred);
		}
		
		System.out.println("Classifying done");
	}
	
	public static void writeToFile(TreeMap<Integer, String> predictedclasses) {
		try {
			FileWriter fw = new FileWriter("run2.txt");
			System.out.println("Writing output to run2.txt");
			for(Map.Entry<Integer, String> classes : predictedclasses.entrySet()) {
				fw.write(classes.getKey() + ".jpg " + classes.getValue() + "\n");
			}
			fw.close(); 
			System.out.println("File is ready");
		} catch(Exception e) {
			System.out.println(e);
		}
	}
	  
}
