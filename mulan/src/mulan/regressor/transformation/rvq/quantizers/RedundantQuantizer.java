package mulan.regressor.transformation.rvq.quantizers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.Random;
import java.util.TreeSet;

import org.apache.commons.math3.util.Combinations;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 * Redundant output space quantizer used in the eMRQ method.
 * 
 * @author Eleftherios Spyromitros-Xioufis
 *
 */
public class RedundantQuantizer extends AbstractQuantizer {

	/**
	 * Number of sub-quantizers to build.
	 */
	private int numQuantizers;

	/**
	 * Number of targets considered in each sub-quantizer.
	 */
	private int NoT;

	/**
	 * Number of centroids in each sub-quantizer.
	 */
	private int NoC;

	/**
	 * This is the maximum number of quantizers allowed.
	 */
	public static final int MAX_NUM_QUANTIZERS = 1000;

	/**
	 * Random seed used in sub-quantizer generation.
	 */
	public static final int RANDOM_SEED = 1;

	/**
	 * 
	 * @param numQuantizers
	 * @param NoT
	 * @param NoC
	 */
	public RedundantQuantizer(int numQuantizers, int NoT, int NoC) {
		this.numQuantizers = numQuantizers;
		this.NoT = NoT;
		this.NoC = NoC;
	}

	@Override
	protected void buildQuantizerInternal(Instances Y) throws Exception {
		quantizers = new ArrayList<Quantizer>(numQuantizers);

		int numTargets = Y.numAttributes();
		if (NoT > numTargets || NoT < 1) {
			throw new Exception("NoT out of range!");
		}

		// find the number of distinct sub-quantizers of size NoT (n!/k!(n-k)!)
		Iterator<int[]> combsIt = new Combinations(numTargets, NoT).iterator();
		int numCombs = 0;
		while (combsIt.hasNext()) {
			combsIt.next();
			numCombs++;
			if (numCombs == MAX_NUM_QUANTIZERS) {
				System.out.println("Number of distinct target combinations reached the maximum number of quantizers.");
				break;
			}
		}
		// System.err.println("Num combinations: " + numCombs);

		// check if number of quantizers requested exceeds the number of distinct sub-quantizers of size NoT
		if (numQuantizers > numCombs) {
			System.err.println("Too many quantizers requested, using " + numCombs);
			numQuantizers = numCombs;
		}

		// implementation of a balanced random selection of numQuantizers target subsets of size NoT
		Random generator = new Random(RANDOM_SEED);
		int[] selectionCounts = new int[numTargets]; // stores the times each target was selected
		ArrayList<TreeSet<Integer>> subsets = new ArrayList<TreeSet<Integer>>(numQuantizers);

		while (subsets.size() < numQuantizers) {
			TreeSet<Integer> selectedTargets = new TreeSet<Integer>();

			for (int j = 0; j < NoT; j++) {
				// find the minimum number of selections
				int min = numQuantizers;
				for (int k = 0; k < numTargets; k++) {
					if (selectionCounts[k] < min) {
						min = selectionCounts[k];
					}
				}

				// add all the targets with minimum number of selections and not already
				// selected in a list
				ArrayList<Integer> selectionCandidates = new ArrayList<Integer>();
				for (int k = 0; k < numTargets; k++) {
					if (selectionCounts[k] == min && !selectedTargets.contains(k)) {
						selectionCandidates.add(k);
					}
				}

				if (selectionCandidates.size() == 0) {
					throw new Exception("No valid candidates available!");
				}

				// shuffle the list and select the first item
				Collections.shuffle(selectionCandidates, generator);
				int selectedTargetIndex = selectionCandidates.get(0);

				// add to selected targets and update the selection counts
				selectedTargets.add(selectedTargetIndex);
				selectionCounts[selectedTargetIndex]++;
			}

			// add the generated subset
			subsets.add(selectedTargets);

		}
		// System.out.println("Subsets generated: " + subsets.toString());

		for (TreeSet<Integer> subset : subsets) {
			// System.out.println("Building quantizer for subset: " + subset.toString());
			Quantizer q = new Quantizer(NoC);

			// keep only target variables whose indices are used in this quantizer
			Remove keepIndices = new Remove();
			int[] selected = new int[subset.size()];
			for (int j = 0; j < selected.length; j++) {
				selected[j] = subset.pollFirst();
			}
			keepIndices.setAttributeIndicesArray(selected);
			keepIndices.setInvertSelection(true);
			keepIndices.setInputFormat(Y);
			Instances Y_selected = Filter.useFilter(Y, keepIndices);

			q.build(Y_selected, selected);

			quantizers.add(q);
		}

	}
}