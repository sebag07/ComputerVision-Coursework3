package uk.ac.soton.ecs.emg;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.sampling.GroupSampler;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.dataset.util.DatasetAdaptors;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.DoubleFVComparison;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.annotation.evaluation.datasets.Caltech101.Record;
import org.openimaj.image.colour.ColourSpace;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.pixel.statistics.HistogramModel;
import org.openimaj.image.processing.convolution.FGaussianConvolve;
import org.openimaj.image.typography.hershey.HersheyFont;
import org.openimaj.math.statistics.distribution.MultidimensionalHistogram;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.util.pair.IntFloatPair;
import org.openimaj.util.pair.Pair;

/**
 * OpenIMAJ Hello world!
 *
 */
public class App {

	public static double getMinValue(double[] array) {
	    double minValue = array[0];
	    for (int i = 1; i < array.length; i++) {
	        if (array[i] < minValue && array[i] != 0.0) {
	            minValue = array[i];
	        }
	    }
	    return minValue;
	}

	public static void main(String[] args) throws FileSystemException {

		GroupedDataset<String, VFSListDataset<FImage>, FImage> training = new VFSGroupDataset<FImage>(
				"D:\\Computer Vision\\Scene_Recognition\\TrainImages\\training", ImageUtilities.FIMAGE_READER);
		GroupedDataset<String, VFSListDataset<FImage>, FImage> testing = new VFSGroupDataset<FImage>(
				"D:\\Computer Vision\\Scene_Recognition\\TestImages\\testing", ImageUtilities.FIMAGE_READER);


		List<FImage> trainingImages = DatasetAdaptors.asList(training);
		List<FImage> testingImages = DatasetAdaptors.asList(testing);

		List<MultidimensionalHistogram> histograms = new ArrayList<MultidimensionalHistogram>();
		HistogramModel model = new HistogramModel(16);

		HashMap<MultidimensionalHistogram, FImage> ImageMap = new HashMap<MultidimensionalHistogram, FImage>();
		HashMap<Double, Pair<FImage>> DistanceMap = new HashMap<Double, Pair<FImage>>();

		for( int i = 0; i < trainingImages.size(); i++ ) {
			FImage img = trainingImages.get(i);
		    model.estimateModel(img);
		    histograms.add(model.histogram.clone());
		    ImageMap.put(model.histogram.clone(), img);
		}

		HistogramModel model2 = new HistogramModel(16);
		FImage testImg = testingImages.get(3);
		model2.estimateModel(testImg);
		MultidimensionalHistogram testHist = model2.histogram;

		double[] distances = new double[trainingImages.size()];
		int count = 0;
		for( int i = 0; i < histograms.size(); i++ ) {
		        double distance = histograms.get(i).compare(testHist, DoubleFVComparison.EUCLIDEAN );
		        DistanceMap.put(distance, new Pair<FImage>(ImageMap.get(histograms.get(i)), testImg));

		        if(distance != 0.0){
					distances[count] = distance;
					count++;
		        }
		}
		System.out.println("distances: " + DistanceMap.size());
		System.out.println("min: " + getMinValue(distances));

		Pair<FImage> minPair = DistanceMap.get(getMinValue(distances));
		DisplayUtilities.display(minPair.firstObject(), "Training Image");
		DisplayUtilities.display(minPair.secondObject(), "Testing Image");
	}
}
