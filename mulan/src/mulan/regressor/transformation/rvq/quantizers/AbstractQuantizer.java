package mulan.regressor.transformation.rvq.quantizers;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;

import weka.core.Instances;

/**
 * All output space quantizers should extend this abstract class.
 * 
 * @author Eleftherios Spyromitros-Xioufis
 *
 */
public abstract class AbstractQuantizer implements Serializable {

	/**
	 * Each output space quantizer consists of one or more {@link Quantizer}s.
	 */
	protected ArrayList<Quantizer> quantizers;

	/**
	 * Abstract build method to be implemented in concrete classes.
	 * 
	 * @param Y the full output space
	 * @throws Exception
	 */
	protected abstract void buildQuantizerInternal(Instances Y) throws Exception;

	/**
	 * Common build method for all output space quantizers. Calls
	 * {@link AbstractQuantizer#buildQuantizerInternal(Instances)} and checks if all target variables have been
	 * quantized.
	 * 
	 * @param Y the full output space
	 * @throws Exception
	 */
	public void buildQuantizer(Instances Y) throws Exception {
		// call the specific build method of each quantizer
		buildQuantizerInternal(Y);

		// check if all target variables have been covered
		int[] selectionCounter = new int[Y.numAttributes()];
		for (Quantizer q : quantizers) {
			int[] indices = q.getIndices();
			for (int index : indices) {
				selectionCounter[index]++;
			}
		}
		System.out.println("Selection counts " + Arrays.toString(selectionCounter));
		for (int i = 0; i < selectionCounter.length; i++) {
			if (selectionCounter[i] == 0) {
				throw new Exception("Target " + i + " hasn't been selected!");
			}
		}

	}

	/**
	 * Returns the sum of squared errors of all the underlying quantizers.
	 * 
	 * @return
	 */
	public double getSquaredError() {
		double sse = 0;
		for (Quantizer q : quantizers) {
			sse += q.getBestQuantizer().getSquaredError();
		}
		return sse;
	}

	/**
	 * Returns the underlying quantizers.
	 * 
	 * @return
	 */
	public ArrayList<Quantizer> getQuantizers() {
		return quantizers;
	}
}
