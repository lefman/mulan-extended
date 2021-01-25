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
package mulan.regressor.transformation;

import mulan.classifier.MultiLabelLearnerBase;
import weka.classifiers.Classifier;
import weka.classifiers.rules.ZeroR;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

/**
 * Base class for multi-target regressors that use a single-target transformation to handle multi-target data.<br>
 * <br>
 * For more information, see:<br>
 * <em>E. Spyromitros-Xioufis, G. Tsoumakas, W. Groves, I. Vlahavas. 2014. Multi-label Classification Methods for
 * Multi-target Regression. <a href="http://arxiv.org/abs/1211.6581">arXiv e-prints</a></em>.
 * 
 * @author Eleftherios Spyromitros-Xioufis
 * @version 2014.04.01
 */
public abstract class TransformationBasedMultiTargetRegressor extends MultiLabelLearnerBase {

    private static final long serialVersionUID = 1L;
    /**
     * The underlying single-target learner.
     */
    protected Classifier baseLearner;

    /**
     * Creates a new instance of {@link TransformationBasedMultiTargetRegressor} with default {@link ZeroR}
     * base regressor.
     */
    public TransformationBasedMultiTargetRegressor() {
        this(new ZeroR());
    }

	/**
	 * Creates a new instance.
	 * 
	 * @param baseLearner
	 *            the base learner which will be used internally to handle the data, not necessarily a regressor
	 */
	public TransformationBasedMultiTargetRegressor(Classifier baseLearner) {
		this.baseLearner = baseLearner;
	}

    /**
     * Returns the {@link Classifier} which is used internally by the learner.
     * 
     * @return the internally used learner
     */
    public Classifier getBaseLearner() {
        return baseLearner;
    }

    /**
     * Returns an instance of a TechnicalInformation object, containing detailed information about the
     * technical background of this class, e.g., paper reference or book this class is based on.
     * 
     * @return the technical information about this class
     */
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result = new TechnicalInformation(Type.INCOLLECTION);
        result.setValue(Field.AUTHOR,
                "Spyromitros-Xioufis, Eleftherios and Tsoumakas, Grigorios and Groves, William and Vlahavas, Ioannis");
        result.setValue(Field.TITLE, "Multi-label Classification Methods for Multi-target Regression");
        result.setValue(Field.JOURNAL, "ArXiv e-prints");
        result.setValue(Field.URL, "http://arxiv.org/abs/1211.6581");
        result.setValue(Field.YEAR, "2014");
        return result;
    }

}
