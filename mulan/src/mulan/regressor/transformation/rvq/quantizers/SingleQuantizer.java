package mulan.regressor.transformation.rvq.quantizers;

import java.util.ArrayList;

import weka.core.Instances;

/**
 * The simplest output space quantization method where a single quantizer is used for the full output space (MRQ
 * method).
 * 
 * @author Eleftherios Spyromitros-Xioufis
 *
 */
public class SingleQuantizer extends AbstractQuantizer {

	/**
	 * Number of centroids of the single output space quantizer.
	 */
	private int numCentroids;

	/**
	 * 
	 * @param numCentroids
	 */
	public SingleQuantizer(int numCentroids) {
		this.numCentroids = numCentroids;
	}

	@Override
	protected void buildQuantizerInternal(Instances Y) throws Exception {
		quantizers = new ArrayList<Quantizer>(1);

		Quantizer q = new Quantizer(numCentroids);
		int[] indices = new int[Y.numAttributes()];
		for (int i = 0; i < Y.numAttributes(); i++) {
			indices[i] = i;
		}
		q.build(Y, indices);
		quantizers.add(q);
	}

}
