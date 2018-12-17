package uk.ac.soton.ecs.sag1g15.run3;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import java.util.*;
import java.util.Map.Entry;

import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.apache.hadoop.hdfs.server.datanode.dataNodeHome_jsp;
import org.hsqldb.jdbc.jdbcBlob;
import org.openimaj.data.*;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.sampling.GroupSampler;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
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
import org.openimaj.image.annotation.evaluation.datasets.Caltech101;
import org.openimaj.image.annotation.evaluation.datasets.Caltech101.Record;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.ml.kernel.HomogeneousKernelMap;
import org.openimaj.ml.kernel.HomogeneousKernelMap.KernelType;
import org.openimaj.ml.kernel.HomogeneousKernelMap.WindowType;
import org.openimaj.ml.annotation.bayes.NaiveBayesAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator.Mode;
import org.openimaj.util.pair.IntFloatPair;

import de.bwaldvogel.liblinear.SolverType;
import eu.larkc.csparql.parser.CSparqlParser.sourceSelector_return;
import eu.larkc.csparql.parser.CSparqlParser.string_return;
import uk.ac.soton.ecs.sag1g15.run2.LinearClassifier;

public class Run3 {
	
	public static void main(String[] args) throws Exception {
		
		File trainingFile = new File("./training/");
		GroupedDataset<String, VFSListDataset<FImage>, FImage> training = new VFSGroupDataset<>(trainingFile.getAbsolutePath(), ImageUtilities.FIMAGE_READER);
		
		File testingFile = new File("./testing/");
		VFSListDataset<FImage> testing = new VFSListDataset<>(testingFile.getAbsolutePath(), ImageUtilities.FIMAGE_READER);
		
		System.out.println(training.size());
		NaiveBayesClassifier bayesClassifier = new NaiveBayesClassifier();
		
		bayesClassifier.train(training);
		
		TreeMap<Integer, String> predictedclasses = new TreeMap<Integer, String>();
		
		printPredictedResults(testing, bayesClassifier, predictedclasses);
		
		writeToFile(predictedclasses); 
		
		bayesClassifier.getResults(training);
	}
	
	public static void printPredictedResults(VFSListDataset<FImage> testing, NaiveBayesClassifier bayesClassifier, TreeMap<Integer, String> predictedclasses) {
		System.out.println("Classifying images");
		
		for(int i = 0; i < testing.size(); i++) {
			ClassificationResult<String> predicted = bayesClassifier.classifyImage(testing.get(i));
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
			FileWriter fw = new FileWriter("run3.txt");
			System.out.println("Writing output to run3.txt");
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