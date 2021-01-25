package mulan.regressor.transformation.rvq;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import mulan.regressor.transformation.rvq.quantizers.AbstractQuantizer;
import mulan.regressor.transformation.rvq.quantizers.Quantizer;
import mulan.regressor.transformation.rvq.quantizers.SingleQuantizer;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Add;
import weka.filters.unsupervised.attribute.Remove;

/**
 * Class that implements the multi-target regression methods described in:<br>
 * 
 * E. Spyromitros-Xioufis, K. Sechidis and I. Vlahavas, "Multi-target regression via output space quantization," 2020
 * International Joint Conference on Neural Networks (IJCNN), Glasgow, United Kingdom, 2020, pp. 1-9, doi:
 * 10.1109/IJCNN48605.2020.9206984.<br>
 * 
 * MRQ, eMRQ and eMRQr are recovered be selecting the appropriate quantization schemes.
 * 
 * @author Eleftherios Spyromitros-Xioufis
 *
 */
public class RegressionViaQuantization extends AbstractRegressionViaQuantization {

	/** Alternative estimation methods */
	public enum EstimationMethod {
		/** Most likely cluster is predicted */
		Normal,
		/** A weighted combination of the most likely clusters is predicted */
		Weighted,
		/** A cost-sensitive prediction is made */
		CostSensitive,
		/** The ground truth is used to predict the best cluster */
		Oracle
	}

	/** Alternative target normalization methods */
	public enum NormalizationMethod {
		/** Leave targets unchanged */
		None,
		/** Normalize targets to 0-1 range before training */
		ZeroOne,
		/** Standardize targets before training */
		Standardize
	}

	/**
	 * The number of stds required for a point to be characterized as outlier. Only used when removeOutliers is true.
	 **/
	public static final double OUTLIER_THRESHOLD = 3;

	/**
	 * Applies 0-1 normalization of the target variables.
	 * 
	 * @param Y         the target variables
	 * @param valuesMin target min values
	 * @param valuesMax target max values
	 * @return
	 */
	public static Instances normalizeTargets(Instances Y, double[] valuesMin, double[] valuesMax) {
		Instances normalized = new Instances(Y);
		for (int j = 0; j < Y.numAttributes(); j++) {
			for (int i = 0; i < Y.numInstances(); i++) {
				double originalValue = Y.instance(i).value(j);
				double normalizedValue = (originalValue - valuesMin[j]) / (valuesMax[j] - valuesMin[j]);
				normalized.instance(i).setValue(j, normalizedValue);
			}
		}

		return normalized;
	}

	/**
	 * Applies standardization of the target variables.
	 * 
	 * @param Y          the target variables
	 * @param valuesMean target means
	 * @param valuesStd  target standard deviations
	 * @return
	 */
	public static Instances standardizeTargets(Instances Y, double[] valuesMean, double[] valuesStd) {
		Instances normalized = new Instances(Y);
		for (int j = 0; j < Y.numAttributes(); j++) {
			for (int i = 0; i < Y.numInstances(); i++) {
				double originalValue = Y.instance(i).value(j);
				double normalizedValue = (originalValue - valuesMean[j]) / valuesStd[j];
				normalized.instance(i).setValue(j, normalizedValue);
			}
		}

		return normalized;
	}

	/** The selected estimation method. */
	protected EstimationMethod estimationMethod;

	/** The selected target normalization method. */
	protected NormalizationMethod normalizationMethod;

	/** Whether to remove outliers before quantization. */
	protected boolean removeOutliers;

	/** Store min and max values of target variables in training set. */
	private double[] valuesMax;
	private double[] valuesMin;
	/** Store means and stds of target variables in training set. */
	private double[] valuesMean;
	private double[] valuesStd;

	/** Number of outliers removed, used for reporting. */
	protected int numOutliersRemoved;

	/** Filter(s) to add new nominal target(s) */
	protected ArrayList<Add> addNewTargets;

	/** The trained multiclass classifier(s) */
	protected ArrayList<Classifier> classifiers;

	/** The used quantizer */
	protected AbstractQuantizer quantizer;

	/**
	 * Default constructor, corresponds to MRQ with k=2 centroids.
	 * 
	 * @param multiclass the underlying multi-class classifier
	 * @throws Exception
	 */
	public RegressionViaQuantization(Classifier multiclass) throws Exception {
		super(multiclass);
		quantizer = new SingleQuantizer(2);
		estimationMethod = EstimationMethod.Normal;
		normalizationMethod = NormalizationMethod.Standardize;
		removeOutliers = false;
	}

	/**
	 * 
	 * @param multiclass the underlying multi-class classifier
	 * @param quantizer  the selected quantization scheme
	 * @throws Exception
	 */
	public RegressionViaQuantization(Classifier multiclass, AbstractQuantizer quantizer) throws Exception {
		super(multiclass);
		this.quantizer = quantizer;
		estimationMethod = EstimationMethod.Normal;
		normalizationMethod = NormalizationMethod.Standardize;
		removeOutliers = false;
	}

	@Override
	protected void buildInternal(MultiLabelInstances trainMT) throws Exception {
		// separate targets from features
		Instances[] XY = separateFeaturesAndTargets(trainMT);
		Instances X = XY[0];
		Instances Y = XY[1];

		// Do normalization and outlier rejection

		// compute and store target statistics that are used in normalization and/or outlier detection
		valuesMax = new double[numLabels];
		valuesMin = new double[numLabels];
		valuesMean = new double[numLabels];
		valuesStd = new double[numLabels];
		for (int j = 0; j < numLabels; j++) {
			valuesMax[j] = Y.attributeStats(j).numericStats.max;
			valuesMin[j] = Y.attributeStats(j).numericStats.min;
			valuesMean[j] = Y.attributeStats(j).numericStats.mean;
			valuesStd[j] = Y.attributeStats(j).numericStats.stdDev;
		}

		boolean[] isOutlier = null;
		if (removeOutliers) { // determine outliers before any normalization of the targets
			isOutlier = new boolean[Y.numInstances()];
			double[] minThresholds = new double[numLabels];
			double[] maxThresholds = new double[numLabels];
			for (int j = 0; j < numLabels; j++) {
				minThresholds[j] = valuesMean[j] - valuesStd[j] * OUTLIER_THRESHOLD;
				maxThresholds[j] = valuesMean[j] + valuesStd[j] * OUTLIER_THRESHOLD;
			}
			for (int i = 0; i < Y.numInstances(); i++) {
				for (int j = 0; j < Y.numAttributes(); j++) {
					if (Y.instance(i).value(j) > maxThresholds[j] || Y.instance(i).value(j) < minThresholds[j]) {
						isOutlier[i] = true;
						break;
					}
				}
			}
		}

		// normalize targets before quantization and learning!
		Instances Y_norm = null;
		switch (normalizationMethod) {
		case None:
			Y_norm = Y;
			break;
		case ZeroOne:
			Y_norm = normalizeTargets(Y, valuesMin, valuesMax);
			break;
		case Standardize:
			Y_norm = standardizeTargets(Y, valuesMean, valuesStd);
			break;
		}

		Instances Y_final = null;
		if (removeOutliers) { // perform outlier removal after normalization if needed
			Y_final = new Instances(Y_norm, 0);
			for (int i = 0; i < Y_norm.numInstances(); i++) {
				if (!isOutlier[i]) {
					Y_final.add(Y_norm.instance(i));
				}
			}
		} else { // just copy all data to Y_final
			Y_final = new Instances(Y_norm);
		}
		numOutliersRemoved = Y_norm.size() - Y_final.size();

		quantizer.buildQuantizer(Y_final);

		ArrayList<Quantizer> quantizers = quantizer.getQuantizers();
		classifiers = new ArrayList<Classifier>(quantizers.size());
		addNewTargets = new ArrayList<Add>(quantizers.size());

		int quantizerIndex = 0;
		for (Quantizer q : quantizers) {
			int[] targetIndices = q.getIndices();
			// System.out.println("Constructing multiclass problem for quantizer: " + quantizerIndex);
			// System.out.println("Target indices: " + Arrays.toString(targetIndices));

			// keep only target variables whose indices are used in this quantizer
			Remove keepIndices = new Remove();
			keepIndices.setAttributeIndicesArray(targetIndices);
			keepIndices.setInvertSelection(true);
			keepIndices.setInputFormat(Y_final);
			Instances Y_selected = Filter.useFilter(Y_final, keepIndices);

			int[] assignments = new int[Y_selected.numInstances()];
			for (int i = 0; i < Y_selected.numInstances(); i++) {
				assignments[i] = q.getBestQuantizer().clusterInstance(Y_selected.instance(i));
			}

			// the following is just a sanity check
			HashSet<Integer> distinctAssignments = new HashSet<Integer>();
			for (int assignment : assignments) {
				distinctAssignments.add(assignment);
			}
			if (distinctAssignments.size() != q.getBestQuantizer().numberOfClusters()) {
				throw new Exception("Unexpected!");
			}

			// use the assignments to create a new attribute that will replace the old one
			// and attach it to the X's
			Add addNewTarget = new Add();
			addNewTarget.setAttributeIndex("last");
			StringBuffer labelList = new StringBuffer();
			HashMap<Integer, Integer> clusterToIndex = new HashMap<Integer, Integer>();
			int order = 0;
			for (int assignment : distinctAssignments) {
				clusterToIndex.put(assignment, order);
				labelList.append(assignment + ",");
				order++;
			}
			addNewTarget.setNominalLabels(labelList.toString().substring(0, labelList.toString().length() - 1));
			addNewTarget.setAttributeName("Clustering " + quantizerIndex);
			addNewTarget.setInputFormat(X);
			Instances X_this = Filter.useFilter(X, addNewTarget);
			X_this.setClassIndex(X_this.numAttributes() - 1); // set this as the class attribute (needed)!
			for (int i = 0; i < X_this.numInstances(); i++) { // 1. nominal index of labels A:0,B:1,C:2,D:3
				X_this.instance(i).setValue(X_this.numAttributes() - 1, clusterToIndex.get(assignments[i]));
			}

			// another sanity check below
			if (X_this.attributeStats(X_this.numAttributes() - 1).distinctCount != q.getBestQuantizer()
					.numberOfClusters()) {
				throw new Exception("Unexpected!");
			}

			// build a multi-class classifier on the transformed dataset
			Classifier multiclass = AbstractClassifier.makeCopy(baseLearner);
			multiclass.buildClassifier(X_this);

			classifiers.add(multiclass);
			addNewTargets.add(addNewTarget);

			quantizerIndex++;
		}
	}

	public int getNumOutliersRemoved() {
		return numOutliersRemoved;
	}

	@Override
	protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception, InvalidDataException {

		// == for oracle starts
		keepY.input(instance);
		double[] gtVals = keepY.output().toDoubleArray();

		// normalize the gt vals if needed!
		for (int i = 0; i < numLabels; i++) {
			switch (normalizationMethod) {
			case None:
				// nothing to do
				break;
			case ZeroOne:
				gtVals[i] = (gtVals[i] - valuesMin[i]) / (valuesMax[i] - valuesMin[i]);
				break;
			case Standardize:
				gtVals[i] = (gtVals[i] - valuesMean[i]) / valuesStd[i];
				break;
			}
		}
		// == for oracle ends
		double[] avgPredVals = new double[numLabels];
		int[] targetPredCounter = new int[numLabels];

		int quantizerIndex = 0;
		for (Quantizer q : quantizer.getQuantizers()) {
			int[] targetIndices = q.getIndices();
			Instances centroids = q.getBestQuantizer().getClusterCentroids();

			double[] predVals = new double[numLabels];
			double[] predCentroid = new double[targetIndices.length];

			if (estimationMethod == EstimationMethod.Oracle) {
				// compute closest centroid to actual values!
				double minDistance = Double.MAX_VALUE;
				int minIndex = 0;
				for (int i = 0; i < centroids.size(); i++) {
					double[] centroid = centroids.get(i).toDoubleArray();
					double distance = 0;
					for (int j = 0; j < targetIndices.length; j++) {
						int gtValsIndex = targetIndices[j];
						distance += (centroid[j] - gtVals[gtValsIndex]) * (centroid[j] - gtVals[gtValsIndex]);
					}
					if (distance < minDistance) {
						minDistance = distance;
						minIndex = i;
					}
				}

				predCentroid = centroids.get(minIndex).toDoubleArray();

			} else {
				Classifier multiclass = classifiers.get(quantizerIndex);

				// transform the given instance to prepare it for the classifier
				keepX.input(instance);
				Instance x = keepX.output(); // only input vector
				Add addNewTarget = addNewTargets.get(quantizerIndex);
				addNewTarget.input(x);
				Instance x_trans = addNewTarget.output(); // with nominal label at the end
				x_trans.dataset().setClassIndex(x_trans.numAttributes() - 1); // setting the class index is needed!

				if (estimationMethod == EstimationMethod.Weighted) {
					// take dist into account to perform a weighted assignment
					double[] dist = multiclass.distributionForInstance(x_trans);
					double sum = Utils.sum(dist);
					if (Math.abs(sum - 1) > 0.01) {
						throw new Exception("Unexpected!");
					}
					for (int i = 0; i < dist.length; i++) {
						double[] centroid = centroids.get(i).toDoubleArray();
						for (int j = 0; j < centroid.length; j++) {
							predCentroid[j] += dist[i] * centroid[j];
						}
					}
				} else if (estimationMethod == EstimationMethod.CostSensitive) {
					// perform a cost-sensitive prediction
					double[] dist = multiclass.distributionForInstance(x_trans);
					double sum = Utils.sum(dist);
					if (Math.abs(sum - 1) > 0.01) {
						throw new Exception("Unexpected!");
					}

					// predict the centroid that incurs the smallest cost
					double minCost = Double.MAX_VALUE;
					int minCostIndex = -1;
					double[][] costMatrix = q.getCostMatrix();

					for (int i = 0; i < dist.length; i++) {
						double cost = 0;
						for (int j = 0; j < dist.length; j++) {
							cost += dist[j] * costMatrix[i][j];
						}
						if (cost < minCost) {
							minCost = cost;
							minCostIndex = i;
						}
					}
					predCentroid = centroids.get(minCostIndex).toDoubleArray();

//					// sanity check
//					int predClusterIndex = (int) multiclass.classifyInstance(x_trans);
//					if (predClusterIndex != minCostIndex) {
//						System.out.println("Difference!");
//					}

				} else if (estimationMethod == EstimationMethod.Normal) {
					int predClusterIndex = (int) multiclass.classifyInstance(x_trans);
					predCentroid = centroids.get(predClusterIndex).toDoubleArray();
				} else {
					throw new Exception("Unknown estimation method!");
				}
			}

			for (int j = 0; j < targetIndices.length; j++) {
				int originalSpaceIndex = targetIndices[j];
				predVals[originalSpaceIndex] = predCentroid[j];
				targetPredCounter[originalSpaceIndex]++;
			}

			for (int j = 0; j < numLabels; j++) {
				avgPredVals[j] += predVals[j];
			}

			quantizerIndex++;
		}

		for (int j = 0; j < numLabels; j++) {
			avgPredVals[j] /= targetPredCounter[j];
		}

		// "de-normalize" targets after prediction!
		for (int i = 0; i < numLabels; i++) {
			switch (normalizationMethod) {
			case None:
				// nothing to do
				break;
			case ZeroOne:
				avgPredVals[i] = avgPredVals[i] * (valuesMax[i] - valuesMin[i]) + valuesMin[i];
				break;
			case Standardize:
				avgPredVals[i] = avgPredVals[i] * valuesStd[i] + valuesMean[i];
				break;
			}
		}

		MultiLabelOutput mlo = new MultiLabelOutput(avgPredVals, true);

		return mlo;
	}

	public void setEstimationMethod(EstimationMethod estimationMethod) {
		this.estimationMethod = estimationMethod;
	}

	public void setNormalizationMethod(NormalizationMethod normalizationMethod) {
		this.normalizationMethod = normalizationMethod;
	}

	public void setRemoveOutliers(boolean removeOutliers) {
		this.removeOutliers = removeOutliers;
	}
}
