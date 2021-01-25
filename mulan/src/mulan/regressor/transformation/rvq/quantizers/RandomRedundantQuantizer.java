package mulan.regressor.transformation.rvq.quantizers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
import java.util.TreeSet;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 * Randomized redundant output space quantizer used in the eMRQr method.
 * 
 * @author Eleftherios Spyromitros-Xioufis
 *
 */
public class RandomRedundantQuantizer extends AbstractQuantizer {

	/**
	 * Number of sub-quantizers to build.
	 */
	private int numQuantizers;

	/**
	 * Range (lower and upper bound) for number of targets considered in each sub-quantizer.
	 */
	private int[] NoTRange;

	/**
	 * Range (lower and upper bound) for number of centroids in each sub-quantizer.
	 */
	private int[] NoCRange;

	/**
	 * Random seed used in sub-quantizer generation.
	 */
	public static final int RANDOM_SEED = 1;

	/**
	 * 
	 * @param numQuantizers
	 * @param NoTRange
	 * @param NoCRange
	 */
	public RandomRedundantQuantizer(int numQuantizers, int[] NoTRange, int[] NoCRange) {
		this.numQuantizers = numQuantizers;
		this.NoTRange = NoTRange;
		this.NoCRange = NoCRange;
	}

	@Override
	protected void buildQuantizerInternal(Instances Y) throws Exception {
		int NoTMin = NoTRange[0];
		int NoTMax = NoTRange[1];

		int NoCMin = NoCRange[0];
		int NoCMax = NoCRange[1];

		int numTargets = Y.numAttributes();

		if (NoTMin < 1 || NoTMax > numTargets || NoTMin > NoTMax) {
			throw new Exception("Check NoT range!");
		}

		if (NoCMin < 2 || NoCMin > NoCMax) {
			throw new Exception("Check NoC range!");
		}

		Random rand = new Random(RANDOM_SEED);

		quantizers = new ArrayList<Quantizer>(numQuantizers);

		int[] selectionCounts = new int[numTargets]; // stores the times each target was selected

		for (int i = 0; i < numQuantizers; i++) {
			// select a random NoT from the range
			int NoT = NoTMin == NoTMax ? NoTMin : NoTMin + rand.nextInt(NoTMax - NoTMin + 1);

			// select a random NoC from the range
			int NoC = NoCMin == NoCMax ? NoCMin : NoCMin + rand.nextInt(NoCMax - NoCMin + 1);

//			 == RANDOM SELECTION ==
//			// randomly select NoT targets without replacement
//			List<Integer> range = IntStream.range(0, numTargets).boxed().collect(Collectors.toList());
//			// System.out.println(range);
//			Collections.shuffle(range, rand);
//			// System.out.prifntln(range);
//			int[] selected = new int[NoT];
//			for (int j = 0; j < NoT; j++) {
//				selected[j] = range.get(j);
//			}

			// == BALANCED RANDOM SELECTION ==
			TreeSet<Integer> selectedTargets = new TreeSet<Integer>();

			for (int j = 0; j < NoT; j++) {
				// find the minimum number of selections
				int min = Integer.MAX_VALUE;
				for (int k = 0; k < numTargets; k++) {
					if (selectionCounts[k] < min) {
						min = selectionCounts[k];
					}
				}

				// add all targets with minimum number of selections and not already selected in a list
				ArrayList<Integer> candidatesForSelection = new ArrayList<Integer>();
				for (int k = 0; k < numTargets; k++) {
					if (selectionCounts[k] == min && !selectedTargets.contains(k)) {
						candidatesForSelection.add(k);
					}
				}

				if (candidatesForSelection.size() == 0) {
					throw new Exception("No valid candidates available!");
				}

				// shuffle the list and select the first item
				Collections.shuffle(candidatesForSelection, rand);
				int selectedTargetIndex = candidatesForSelection.get(0);

				// add to selected targets and update the selection counts
				selectedTargets.add(selectedTargetIndex);
				selectionCounts[selectedTargetIndex]++;
			}

			int[] selected = new int[selectedTargets.size()];
			for (int j = 0; j < selected.length; j++) {
				selected[j] = selectedTargets.pollFirst();
			}

			// keep only target variables whose indices are used in this quantizer
			Remove keepIndices = new Remove();
			keepIndices.setAttributeIndicesArray(selected);
			keepIndices.setInvertSelection(true);
			keepIndices.setInputFormat(Y);
			Instances Y_selected = Filter.useFilter(Y, keepIndices);

//			System.out.println("NoT: " + NoT + " NoC: " + NoC);
//			System.out.println(Arrays.toString(selected));

			Quantizer q = new Quantizer(NoC);
			q.build(Y_selected, selected);

			quantizers.add(q);

		}

	}

}
