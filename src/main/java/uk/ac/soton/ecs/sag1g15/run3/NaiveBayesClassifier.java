package uk.ac.soton.ecs.sag1g15.run3;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.sampling.GroupSampler;
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
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.image.feature.local.aggregate.PyramidSpatialAggregator;
import org.openimaj.ml.annotation.bayes.NaiveBayesAnnotator;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.ml.kernel.HomogeneousKernelMap;
import org.openimaj.ml.kernel.HomogeneousKernelMap.KernelType;
import org.openimaj.ml.kernel.HomogeneousKernelMap.WindowType;
import org.openimaj.util.pair.IntFloatPair;

public class NaiveBayesClassifier {
	
	private NaiveBayesAnnotator<FImage, String> ann;
	GroupedDataset<String, ListDataset<FImage>, FImage> data;
	GroupedRandomSplitter<String, FImage> splits;
	
	
	public void train(GroupedDataset<String, VFSListDataset<FImage>, FImage> training) throws FileSystemException {
		
		
		data = GroupSampler.sample(training, 5, false);

		splits = new GroupedRandomSplitter<String, FImage>(data, 15, 0, 0);
		
		DenseSIFT dsift = new DenseSIFT(3, 7);
		PyramidDenseSIFT<FImage> pdsift = new PyramidDenseSIFT<FImage>(dsift, 6f, 7);
		
		HardAssigner<byte[], float[], IntFloatPair> assigner = trainQuantiser(splits.getTrainingDataset(), pdsift);
	
		HomogeneousKernelMap hKernelMap = new HomogeneousKernelMap(KernelType.Chi2, WindowType.Rectangular);
		
		FeatureExtractor<DoubleFV, FImage> extractor = hKernelMap.createWrappedExtractor(new PHOWExtractor(pdsift, assigner));
		
		ann = new NaiveBayesAnnotator<FImage, String>(
	            extractor, NaiveBayesAnnotator.Mode.MAXIMUM_LIKELIHOOD);
		System.out.println("Start training");
		ann.train(splits.getTrainingDataset());
		System.out.println("Training done");
		
	}
	
	public ClassificationResult<String> classifyImage(FImage image){
		return ann.classify(image);
	}
	
	public void getResults(GroupedDataset<String, VFSListDataset<FImage>, FImage> training){

		splits = new GroupedRandomSplitter<String, FImage>(data, 15, 0, 15);
		
		ClassificationEvaluator<CMResult<String>, String, FImage> eval = 
		new ClassificationEvaluator<CMResult<String>, String, FImage>(
			ann, splits.getTrainingDataset(), new CMAnalyser<FImage, String>(CMAnalyser.Strategy.SINGLE));
			
		Map<FImage, ClassificationResult<String>> guesses = eval.evaluate();
		CMResult<String> result = eval.analyse(guesses);

		System.out.println(result.getDetailReport());
	}
	

	public HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(GroupedDataset<String, ListDataset<FImage>, FImage> groupedDataset, PyramidDenseSIFT<FImage> pdsift){
		
		 List<LocalFeatureList<ByteDSIFTKeypoint>> allkeys = new ArrayList<LocalFeatureList<ByteDSIFTKeypoint>>();
		
		 System.out.println("Sift train has started");
		 
		 for(FImage rec : groupedDataset) {
			 FImage image = rec.getImage();
			 
			 pdsift.analyseImage(image);
			 allkeys.add(pdsift.getByteKeypoints(0.005f));
		 }
		 
		 System.out.println("Sift train has ended");
		 
		 if(allkeys.size() > 10000) {
			 allkeys = allkeys.subList(0, 10000);
		 }
		 
		 ByteKMeans kMeans = ByteKMeans.createKDTreeEnsemble(600);
		 DataSource<byte[]> datasource = new LocalFeatureListDataSource<ByteDSIFTKeypoint, byte[]>(allkeys);
		 System.out.println("Clustering has started");
		 ByteCentroidsResult result = kMeans.cluster(datasource);
		 System.out.println("Clustering done");
		 
		 return result.defaultHardAssigner();
	}
	
	class PHOWExtractor implements FeatureExtractor<DoubleFV, FImage> {

		PyramidDenseSIFT<FImage> pdsift;
		HardAssigner<byte[], float[], IntFloatPair> assigner;
		
		public PHOWExtractor(PyramidDenseSIFT<FImage> pdsift,
		HardAssigner<byte[], float[], IntFloatPair> assigner) {
			this.pdsift = pdsift;
			this.assigner = assigner;
		}
		
		@Override
		public DoubleFV extractFeature(FImage object) {
			FImage image = object.getImage();
			pdsift.analyseImage(image);
			
			BagOfVisualWords<byte[]> bagOfVisualWords = new BagOfVisualWords<byte[]>(assigner);
		
			PyramidSpatialAggregator<byte[], SparseIntFV> spatial = new PyramidSpatialAggregator<byte[], SparseIntFV>(bagOfVisualWords, 2, 4);
		
			return spatial.aggregate(pdsift.getByteKeypoints(0.015f), image.getBounds()).normaliseFV();
			
		}
		
	}
	
}
