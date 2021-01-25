package mulan.evaluation.measures.regression.example;

import mulan.classifier.MultiLabelOutput;

/**
 * Computes the example-based squared error.
 * 
 * @author Eleftherios Spyromitros-Xioufis
 * @version 2014.11.07
 */
public class ExampleBasedSE extends ExampleBasedRegressionMeasureBase {

	public String getName() {
		return "Example-based SE";
	}

	public double getIdealValue() {
		return 0;
	}

	public void updateInternal(MultiLabelOutput prediction, double[] truth) {
		double[] scores = prediction.getPvalues();
		double squaredError = 0;
		for (int i = 0; i < truth.length; i++) {
			if (Double.isNaN(truth[i])) {
				continue; // error on missing targets is considered 0
			}
			squaredError += (truth[i] - scores[i]) * (truth[i] - scores[i]);
		}
		sum += squaredError;
		count++;

	}
}
