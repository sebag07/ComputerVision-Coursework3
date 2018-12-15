package uk.ac.soton.ecs.sag1g15.run2;

import java.util.*;

import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FloatFV;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.LocalFeature;
import org.openimaj.feature.local.LocalFeatureImpl;
import org.openimaj.feature.local.SpatialLocation;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.image.pixel.sampling.RectangleSampler;
import org.openimaj.math.geometry.shape.Rectangle;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator.Mode;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.array.ArrayUtils;
import org.openimaj.util.pair.IntFloatPair;

import de.bwaldvogel.liblinear.SolverType;

public class LinearClassifier {
	
	public int clusters = 500;
	public int count = 0;
	public int size = 8;
	public int step = 4;
	
	
	private LiblinearAnnotator<FImage, String> annotator;
	
	
	public List<LocalFeature<SpatialLocation, FloatFV>> extract(FImage image, float step, float size){
		
		
		System.out.println("Extracting patches: " + count);
		count++;
		ArrayList<LocalFeature<SpatialLocation, FloatFV>> featureList = new ArrayList<LocalFeature<SpatialLocation, FloatFV>>();
		
		RectangleSampler rectangleSampler = new RectangleSampler(image, step, step, size, size);

		for(Rectangle rectangle : rectangleSampler) {
			FImage patch = image.extractROI(rectangle);
			
			LocalFeature<SpatialLocation, FloatFV> localFeature = getPatchFeatures(patch, rectangle);
			
			featureList.add(localFeature);
		}
		
		return featureList;
	}
	
	public LocalFeature<SpatialLocation, FloatFV> getPatchFeatures(FImage patch, Rectangle rectangle){
		float[] vector = ArrayUtils.reshape(patch.pixels);
		FloatFV featureVector = new FloatFV(vector);
		
		SpatialLocation spatialLocation = new SpatialLocation(rectangle.x, rectangle.y);
		
		return new LocalFeatureImpl<SpatialLocation, FloatFV>(spatialLocation, featureVector);	
	}
	
	public HardAssigner<float[], float[], IntFloatPair> quantisy(Dataset<FImage> sample){
		
		System.out.println("Am intrat in HardAssigner");
		
		ArrayList<float[]> allkeys = new ArrayList<float[]>();
		
		for(FImage image : sample) {
			List<LocalFeature<SpatialLocation, FloatFV>> sampleList = extract(image, step, size);
			
			for(LocalFeature<SpatialLocation, FloatFV> localFeature : sampleList) {
				allkeys.add(localFeature.getFeatureVector().values);
			}
		}
		
		FloatCentroidsResult results = generateClusters(clusters, allkeys);
		
		System.out.println("Clusters have been generated");
		
		return results.defaultHardAssigner();
	}
	
	public FloatCentroidsResult generateClusters(int clusters, ArrayList<float[]> allkeys) {
		FloatKMeans kMeans = FloatKMeans.createKDTreeEnsemble(clusters);
		float[][] array = allkeys.toArray(new float[][] {});
		
		return kMeans.cluster(array);
	}
	
	public void trainImages(GroupedDataset<String, VFSListDataset<FImage>, FImage> training) {
		
		System.out.println("Training image");
		
		GroupedRandomSplitter<String, FImage> randomSplitter = new GroupedRandomSplitter<String, FImage>(training, 15, 0, 0);
		HardAssigner<float[], float[], IntFloatPair> assigner = quantisy(randomSplitter.getTrainingDataset());
		
		System.out.println("HardAssigner assigned");
		
		PatchClusterFeatureExtractor featureExtractor = new PatchClusterFeatureExtractor(assigner);
		System.out.println("Started training...");
		annotator = new LiblinearAnnotator<FImage, String>(featureExtractor, Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
		annotator.train(randomSplitter.getTrainingDataset());
		
		System.out.println("Images have been trained");
	}
	
	public LiblinearAnnotator getAnnotator() {
		return this.annotator;
	}
	
	
	class PatchClusterFeatureExtractor implements FeatureExtractor<DoubleFV, FImage> {

		HardAssigner<float[], float[], IntFloatPair> assigner;
		
		public PatchClusterFeatureExtractor(HardAssigner<float[], float[], IntFloatPair> assigner) {
			this.assigner = assigner;
		}
		
		@Override
		public DoubleFV extractFeature(FImage patch) {
			
			BagOfVisualWords<float[]> bagOfVisualWords = new BagOfVisualWords<float[]>(assigner);
			
			BlockSpatialAggregator<float[], SparseIntFV> spatialAggregator = new BlockSpatialAggregator<float[], SparseIntFV>(bagOfVisualWords, 2, 2);
			
			return spatialAggregator.aggregate(extract(patch, step, size), patch.getBounds()).normaliseFV();
		}
		
	}
}
