package mulan.regressor.transformation.rvq.quantizers;

import java.util.Arrays;
import java.util.HashSet;

import weka.clusterers.AbstractClusterer;
import weka.clusterers.SimpleKMeans;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;

/**
 * This class serves as the basic building block for creating more complex quantizers. It uses Weka's SimpleKMeans
 * internally but adapts its functionality to the needs of Multi-target Regression via Quantization methods.
 * 
 * @author Eleftherios Spyromitros-Xioufis
 *
 */
public class Quantizer {

	/** Requested number of clusters **/
	protected int numClustersRequested;

	/** Whether to use the k-means++ initialization method */
	protected boolean useKmeansPlusPlus;

	/** Number of k-means restarts with random initialization to select the quantizer with lowest error */
	protected int numRestarts;

	/** Maximum number of k-means iterations **/
	protected int numKmeansIters;

	/** The starting random seed to use. Starting seed is incremented by 1 when multiple quantizers are tried. */
	protected int startingSeed;

	/** The average (per Instance) quantization error (Squared Error) in the training set. */
	protected double trainSEPerInstance;

	/** The target indices this quantizer is about **/
	protected int[] indices;

	/** The learned quantizer */
	protected SimpleKMeansExtended quantizer;

	/**
	 * A symmetric k x k matrix (k is the number of centroids in the learned quantizer) where each off diagonal element
	 * [i][j] contain the squared distance between the i-th and j-th centroid.
	 * 
	 * cost-matrix generated based on distances between centroids and within cluster distances
	 */
	protected double[][] costMatrix;

	/** Default number of k-means iterations. */
	public static final int DEFAULT_NUM_ITERS = 500;

	/** Default number of k-means restarts. */
	public static final int DEFAULT_NUM_RESTARTS = 1;

	/** Default starting random seed. */
	public static final int DEFAULT_STARTING_SEED = 0;

	/** Making this variable static because parallel k-means is unstable, i.e. not repeatable */
	public static final int NUM_SLOTS = 1;

	/** Whether to use within cluster distances to populate diagonal elements of {@link Quantizer#costMatrix} */
	public static final boolean USE_WITHIN_CLUSTER_DISTANCES_IN_COST_MATRIX = false;

	/** Whether to output debug messages */
	protected static final boolean DEBUG = false;

	/** Default starting random seed */
	public static final boolean DEFAULT_USE_KMEANS_PLUS_PLUS = true;

	public Quantizer(int numClusters) {
		this.numClustersRequested = numClusters;

		this.startingSeed = DEFAULT_STARTING_SEED;
		this.numKmeansIters = DEFAULT_NUM_ITERS;
		this.numRestarts = DEFAULT_NUM_RESTARTS;
		this.useKmeansPlusPlus = DEFAULT_USE_KMEANS_PLUS_PLUS;
	}

	public Quantizer(int numClusters, boolean useKmeansPlusPlus, int numRestarts) {
		this.numClustersRequested = numClusters;
		this.useKmeansPlusPlus = useKmeansPlusPlus;
		this.numRestarts = numRestarts;

		this.startingSeed = DEFAULT_STARTING_SEED;
		this.numKmeansIters = DEFAULT_NUM_ITERS;
	}

	/**
	 * 
	 * @param Y       the dataset (set of variables) on which the quantizer will be built
	 * @param indices the indices of variables (in original target space) on which the quantizer will be built
	 * @throws Exception
	 */
	public void build(Instances Y, int[] indices) throws Exception {
		if (Y.numAttributes() != indices.length) {
			throw new Exception("Indices array size should be equal to Y dimensionality!");
		}

		this.indices = indices;

		// compute number of distinct target combinations so that an appropriate number of clusters is requested
		HashSet<String> distinctCombs = new HashSet<String>();
		for (int i = 0; i < Y.numInstances(); i++) {
			String combination = "";
			for (int j = 0; j < Y.numAttributes(); j++) {
				combination += Y.instance(i).value(j) + ",";
			}
			distinctCombs.add(combination);
		}
		int targetSpaceCardinality = distinctCombs.size();
		// there is no point in requesting more clusters than the cardinality of the target space
		int numClustersToCreate = Math.min(numClustersRequested, targetSpaceCardinality);

		// apply k-means on Y
		if (DEBUG) {
			System.out.println("Clusters requested :" + numClustersRequested);
		}
		quantizer = buildInternal(Y, numClustersToCreate, numRestarts, startingSeed, useKmeansPlusPlus, numKmeansIters);
		if (DEBUG) {
			System.out.println("Clusters created :" + quantizer.getNumClusters());
		}

		// This is the total squared error of the quantizer (in the training set)
		double SE = quantizer.getSquaredError();
		if (DEBUG) {
			System.err.println("SE: " + SE + " Seed: " + quantizer.getSeed());
		}
		// divide by the number of training data to get average squared error per instance
		trainSEPerInstance = SE / Y.numInstances();

		// store centroids of the quantizer
		Instances centroids = quantizer.getClusterCentroids();
		// standard deviations can be used in cost-matrix generation
		// Instances stds = quantizer.getClusterStandardDevs();

		if (numClustersToCreate != centroids.size()) {
			// it is expected than in some cases, k-means will generate less clusters than requested, because some
			// initial clusters end up becoming empty
			if (DEBUG) {
				System.out.println("Clusters requested/tried/generated: " + numClustersRequested + "/"
						+ numClustersToCreate + "/" + centroids.size());
			}

		}

		// generate a cost-matrix for this quantizer
		costMatrix = new double[centroids.numInstances()][centroids.numInstances()];
		for (int i = 0; i < centroids.numInstances(); i++) {
			for (int j = 0; j <= i; j++) {
				if (i == j) {
					if (USE_WITHIN_CLUSTER_DISTANCES_IN_COST_MATRIX) { // use within cluster SSE
						costMatrix[i][j] = quantizer.getSquaredErrors()[i] / quantizer.getClusterSizes()[i];
					} else { // set to 0
						costMatrix[i][j] = 0;
					}
				} else {
					// compute distance between centroids
					Instance centroid1 = centroids.get(i);
					Instance centroid2 = centroids.get(j);
					for (int k = 0; k < centroid1.numAttributes(); k++) {
						costMatrix[i][j] += (centroid1.value(k) - centroid2.value(k))
								* (centroid1.value(k) - centroid2.value(k));

					}
					costMatrix[j][i] = costMatrix[i][j];
				}

			}
		}

	}

	/**
	 * Helper method to build quantizers. Builds a k-means quantizer with {@link Quantizer#numClustersRequested} on the
	 * supplied {@code data}. If {@link Quantizer#numRestarts} > 1 the method builds multiple quantizers using different
	 * random seeds ({@link Quantizer#startingSeed},
	 * {@link Quantizer#startingSeed}+1,...,{@link Quantizer#startingSeed}+{@link Quantizer#numRestarts}-1) and returns
	 * the one with the lowest squared error.
	 * 
	 * @param data                 the data used to build the quantizer
	 * @param numClustersRequested the requested number of clusters (resulting number of clusters may be less)
	 * @param numRestarts          number of randomly initialized quantizers to try
	 * @param startingSeed         the random seed to start with when building quantizers
	 * @param maxIters             the maximum number of iterations in k-means
	 * @return
	 * @throws Exception
	 */
	public static SimpleKMeansExtended buildInternal(Instances data, int numClustersRequested, int numRestarts,
			int startingSeed, boolean useKmeansPlusPlus, int maxIters) throws Exception {
		if (numRestarts <= 0) {
			throw new Exception("Num quantizers should be > 0.");
		}

		double minSSE = Double.MAX_VALUE;
		SimpleKMeansExtended bestQuantizer = null;
		// run k-means with different random seeds and keep the one with lowest SSE
		for (int i = startingSeed; i < startingSeed + numRestarts; i++) {
			// Create a new k-means instance
			SimpleKMeansExtended kmeans = new SimpleKMeansExtended();
			kmeans.setNumExecutionSlots(NUM_SLOTS);
			kmeans.setMaxIterations(maxIters);
			kmeans.setNumClusters(numClustersRequested);
			if (useKmeansPlusPlus) {
				kmeans.setInitializationMethod(
						new SelectedTag(SimpleKMeans.KMEANS_PLUS_PLUS, SimpleKMeans.TAGS_SELECTION));
			}
			// !!! Turning normalization off here is important because we experiment with alternative explicit
			// normalizations of the data.
			EuclideanDistance ed = new EuclideanDistance();
			ed.setDontNormalize(true);
			kmeans.setDistanceFunction(ed);

			kmeans.setDisplayStdDevs(true);
			kmeans.setSeed(i);
			kmeans.setPreserveInstancesOrder(true); // needed because we need the assignments!
			kmeans.setDebug(DEBUG);

			if (DEBUG) {
				System.out.println(Arrays.toString(kmeans.getOptions()));
			}

			// build the quantizer
			kmeans.buildClusterer(data);

			double SSE = kmeans.getSquaredError();
			if (DEBUG) {
				System.out.println("Run " + (i + 1) + " SSE: " + SSE);
			}
			if (SSE < minSSE) {
				minSSE = SSE;
				bestQuantizer = (SimpleKMeansExtended) AbstractClusterer.makeCopy(kmeans);
			}
		}
		if (DEBUG) {
			System.out.println("Num clusters created: " + bestQuantizer.numberOfClusters());
		}

		return bestQuantizer;
	}

	public int getNumClustersGenerated() {
		return quantizer.getClusterCentroids().size();
	}

	public int[] getIndices() {
		return indices;
	}

	public void setIndices(int[] indices) {
		this.indices = indices;
	}

	public double[][] getCostMatrix() {
		return costMatrix;
	}

	public SimpleKMeansExtended getBestQuantizer() {
		return quantizer;
	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub
	}

}
