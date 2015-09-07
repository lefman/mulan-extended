/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */
package mulan.evaluation.measure;

import java.util.Arrays;

import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;

/**
 * Implementation of the average Relative Root Mean Squared Error (RRMSE) measure. RRMSE for each
 * targe is equal to the RMSE of the prediction divided by the RMSE of predicting the mean. Two
 * versions of this measure are implemented, one computes target means on the training set and the
 * other on the union of the training set and the test set. The first version is the default one.
 * 
 * @author Eleftherios Spyromitros-Xioufis
 * @version 2013.07.26
 */
public class AverageRelativeRMSE extends AverageRMSE implements MacroAverageMeasure {

    /** holds the mean (calculated from train set) prediction's total squared error per target */
    private double[] trainMeanPredTotalSquaredError;
    /** holds the mean (calculated from full set) prediction's total squared error per target */
    private double[] fullMeanPredTotalSquaredError;
    /** holds the mean per target in train set */
    private double[] targetMeansTrain;
    /** holds the mean per target in full dataset */
    private double[] targetMeansFull;

    public AverageRelativeRMSE(int numOfLabels, MultiLabelInstances train, MultiLabelInstances test) {
        super(numOfLabels);
        trainMeanPredTotalSquaredError = new double[numOfLabels];
        fullMeanPredTotalSquaredError = new double[numOfLabels];
        targetMeansTrain = new double[numOfLabels];
        targetMeansFull = new double[numOfLabels];
        int[] labelIndices = train.getLabelIndices();
        for (int i = 0; i < numOfLabels; i++) {
            targetMeansTrain[i] = train.getDataSet().attributeStats(labelIndices[i]).numericStats.mean;
            double testAverage = test.getDataSet().attributeStats(labelIndices[i]).numericStats.mean;
            int trainInstances = train.getDataSet().numInstances();
            int testInstances = test.getDataSet().numInstances();
            int allInstances = trainInstances + testInstances;
            targetMeansFull[i] = (targetMeansTrain[i] * trainInstances + testAverage
                    * testInstances)
                    / allInstances;
        }
    }

    public String getName() {
        return "Average Relative RMSE";
    }

    /**
     * Returns the value of the measure for each target
     * 
     * @param targetIndex the index of the target
     * @return the value of the measure
     */
    @Override
    public double getValue(int targetIndex) {
        double mse = totalSquaredError[targetIndex] / nonMissingCounter[targetIndex];
        double rel_mse = trainMeanPredTotalSquaredError[targetIndex]
                / nonMissingCounter[targetIndex];
        double root_mse = Math.sqrt(mse);
        double root_rel_mse = Math.sqrt(rel_mse);
        double rrmse = root_mse / root_rel_mse;
        return rrmse;
    }

    public double getTotalSE(int labelIndex) {
        double mse = totalSquaredError[labelIndex];
        return mse;
    }

    public double getTrainMeanTotalSE(int labelIndex) {
        double mse = trainMeanPredTotalSquaredError[labelIndex];
        return mse;
    }

    public double getFullMeanTotalSE(int labelIndex) {
        double mse = fullMeanPredTotalSquaredError[labelIndex];
        return mse;
    }

    public double getTargetAverageFull(int labelIndex) {
        return targetMeansFull[labelIndex];
    }

    public double getTargetAverageTrain(int labelIndex) {
        return targetMeansTrain[labelIndex];
    }

    /**
     * When a target has missing values, they are ignored in RRMSE calculation.
     */
    @Override
    protected void updateInternal(MultiLabelOutput prediction, double[] truth) {
        double[] scores = prediction.getPvalues();
        for (int i = 0; i < truth.length; i++) {
            if (Double.isNaN(truth[i])) {
                continue;
            }
            nonMissingCounter[i]++;
            totalSquaredError[i] += (truth[i] - scores[i]) * (truth[i] - scores[i]);
            trainMeanPredTotalSquaredError[i] += (truth[i] - targetMeansTrain[i])
                    * (truth[i] - targetMeansTrain[i]);
            fullMeanPredTotalSquaredError[i] += (truth[i] - targetMeansFull[i])
                    * (truth[i] - targetMeansFull[i]);
        }
    }

    @Override
    public void reset() {
        super.reset();
        Arrays.fill(trainMeanPredTotalSquaredError, 0.0);
        Arrays.fill(fullMeanPredTotalSquaredError, 0.0);

    }
}
