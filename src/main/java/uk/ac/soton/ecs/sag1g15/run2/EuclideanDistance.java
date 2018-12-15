package uk.ac.soton.ecs.sag1g15.run2;

import java.util.Comparator;

import org.openimaj.feature.FloatFV;
import org.openimaj.feature.FloatFVComparison;

public class EuclideanDistance implements Comparator<FloatFV> {

	FloatFV vector;
	
	public void setVector(FloatFV vector) {
		this.vector = vector;
	}
	
	@Override
	public int compare(FloatFV o1, FloatFV o2) {
		FloatFVComparison comparator = FloatFVComparison.EUCLIDEAN;
		Double distance0 = comparator.compare(o1, vector);
		Double distance1 = comparator.compare(o2, vector);
		return distance0.compareTo(distance1);
	}

}
