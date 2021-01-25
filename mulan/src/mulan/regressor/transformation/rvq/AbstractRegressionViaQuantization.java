package mulan.regressor.transformation.rvq;

import mulan.data.MultiLabelInstances;
import mulan.regressor.transformation.TransformationBasedMultiTargetRegressor;
import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 * Class that abstracts common functionality for all methods that perform multi-target regression via quantization
 * described in:<br>
 * 
 * E. Spyromitros-Xioufis, K. Sechidis and I. Vlahavas, "Multi-target regression via output space quantization," 2020
 * International Joint Conference on Neural Networks (IJCNN), Glasgow, United Kingdom, 2020, pp. 1-9, doi:
 * 10.1109/IJCNN48605.2020.9206984.
 * 
 * @author Eleftherios Spyromitros-Xioufis
 *
 */
public abstract class AbstractRegressionViaQuantization extends TransformationBasedMultiTargetRegressor {

	/** Filter to keep only targets */
	protected Remove keepY;
	/** Filter to keep only features */
	protected Remove keepX;

	public AbstractRegressionViaQuantization(Classifier multiclass) {
		super(multiclass);
	}

	/**
	 * Initializes the remove filters to keep features/targets and returns the filtered features and targets.
	 * 
	 * @param trainMT
	 * @return
	 * @throws Exception
	 */
	public Instances[] separateFeaturesAndTargets(MultiLabelInstances trainMT) throws Exception {
		// first select only the target attributes
		Instances train = trainMT.getDataSet();
		keepY = new Remove();
		keepY.setAttributeIndicesArray(trainMT.getFeatureIndices());
		keepY.setInputFormat(train);
		Instances Y = Filter.useFilter(train, keepY);

		// rem is now ready for filtering test instances
		keepX = new Remove();
		keepX.setAttributeIndicesArray(trainMT.getFeatureIndices());
		keepX.setInvertSelection(true);
		keepX.setInputFormat(train);
		Instances X = Filter.useFilter(train, keepX);
		Instances[] XY = { X, Y };
		return XY;
	}

}
