package uk.ac.soton.ecs.sag1g15.run2;

import java.util.List;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.util.DatasetAdaptors;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;

public class Run2 {

	public static void main(String[] args) throws FileSystemException {
		LinearClassifier linearClassifier = new LinearClassifier();
		
		VFSListDataset<FImage> testing = new VFSListDataset<>("C:\\Users\\tzica\\Coursework3\\testing", ImageUtilities.FIMAGE_READER);
		
		GroupedDataset<String, VFSListDataset<FImage>, FImage> training = new VFSGroupDataset<>("C:\\Users\\tzica\\Coursework3\\training", ImageUtilities.FIMAGE_READER);
		
		List<FImage> testingImages = DatasetAdaptors.asList(testing);
		
		linearClassifier.trainImages(training);
		
		System.out.println(linearClassifier.getAnnotator().classify(testingImages.get(3)));
	}
	  
}
