PROJECT REPORT
Portuguese Bank Term Deposit Prediction
PRCP-1000 | Marketing Campaign Analytics & Machine Learning

A comprehensive machine learning study to predict customer subscription to term deposits through direct phone-call marketing campaign data, aimed at maximizing campaign ROI for a Portuguese banking institution.

Project ID:	PRCP-1000-PortugeseBank
Industry:	Banking & Financial Services
Technique:	Supervised Binary Classification
Best Model:	Random Forest Classifier — 86.3% Accuracy
 
1. Executive Summary
This report documents the end-to-end data science project undertaken for a Portuguese banking institution to optimize its direct phone-call marketing campaigns. The project analyzed 11,162 historical customer interaction records from prior campaigns, encompassing 17 attributes covering customer demographics, financial status, previous campaign history, and call characteristics. The primary deliverable is a predictive machine learning model that identifies customers most likely to subscribe to a term deposit product, enabling the bank to focus its marketing spend on the highest-probability prospects.

Six classification models were trained, evaluated, and compared: Logistic Regression, GridSearchCV-tuned Decision Tree, Decision Tree, Random Forest, AdaBoost, and a Voting Classifier ensemble. Random Forest Classifier emerged as the best-performing model with an accuracy of 86.3% and the fewest misclassifications (308 out of 2,233 test observations), and is the recommended model for production deployment.

Beyond model selection, the project produced four actionable business recommendations directly guiding campaign strategy: optimal call duration windows, customer balance thresholds, age group targeting, and monthly timing for outreach.

Key Result: Random Forest Classifier achieved 86.3% accuracy on the test set — the highest among all six models — with only 308 misclassifications out of 2,233 test records. The top predictive features were: call duration, customer account balance, customer age, and day of contact.

2. Project Overview
2.1 Business Context
The Portuguese banking institution conducted direct marketing campaigns to promote term deposit subscriptions (fixed-deposit products) via outbound telephone calls. These campaigns are resource-intensive, requiring agent time per call, and their effectiveness is highly dependent on which customers are targeted. Historical campaign data shows a deposit subscription rate of approximately 47.4% (YES: 5,289 out of 11,162 total customers), indicating that roughly half of all contacted customers do not subscribe — representing a significant cost inefficiency.

By predicting in advance which customers are likely to subscribe, the bank can concentrate outreach resources on high-probability segments, reduce wasted calls to unlikely subscribers, and ultimately maximize the return on investment (ROI) of its marketing operations.

2.2 Project Objectives
•	Perform comprehensive Exploratory Data Analysis (EDA) to understand feature distributions and relationships with the target variable (deposit subscription).
•	Clean and preprocess the dataset including outlier analysis, feature encoding, and dimensionality decisions.
•	Build, evaluate, and compare six machine learning classification models.
•	Select the best model based on accuracy and confusion matrix analysis.
•	Extract actionable business recommendations from feature importance analysis.
•	Provide a production deployment roadmap using AWS SageMaker.

2.3 Dataset Summary
Attribute	Detail
Dataset Name	bank.csv — Portuguese Bank Marketing Campaign
Total Records	11,162 customer rows
Total Features	17 columns (16 predictors + 1 target)
Target Variable	deposit (binary: yes / no — term deposit subscription)
Class Distribution	YES: 5,289 (47.4%)  |  NO: 5,873 (52.6%)
Numerical Features	7 (age, balance, day, duration, campaign, pdays, previous)
Categorical Features	10 (job, marital, education, default, housing, loan, contact, month, poutcome, deposit)
Missing Values	None (confirmed)
Duplicate Rows	Zero duplicates detected

2.4 Feature Dictionary
#	Feature	Type	Description
1	age	Integer	Age of the customer (range: 18–95, mean: 41.23)
2	job	Categorical	Type of occupation (12 unique categories)
3	marital	Categorical	Marital status: married, single, divorced
4	education	Categorical	Education level: primary, secondary, tertiary, unknown
5	default	Binary	Has the customer defaulted on a loan? (yes/no)
6	balance	Integer	Account balance in euros (range: -6,847 to 81,204; mean: 1,528.53)
7	housing	Binary	Has a housing loan? (yes/no)
8	loan	Binary	Has a personal loan? (yes/no)
9	contact	Categorical	Contact communication type: cellular, telephone, unknown
10	day	Integer	Last contact day of the month (1–31)
11	month	Categorical	Last contact month of the year
12	duration	Integer	Last contact call duration in seconds
13	campaign	Integer	Number of contacts during this campaign for this customer
14	pdays	Integer	Days since last contact from previous campaign (-1 = not contacted)
15	previous	Integer	Number of contacts before this campaign
16	poutcome	Categorical	Previous campaign outcome: success, failure, unknown
17	deposit (Target)	Binary	Subscribed to term deposit? YES / NO

3. Problem Statement
A Portuguese banking institution conducts direct marketing campaigns via telephone to solicit subscriptions to term deposit products. These campaigns require significant investment in agent staffing, call infrastructure, and campaign management. The fundamental inefficiency is that the bank contacts a broad customer base indiscriminately, leading to a high proportion of unproductive calls where customers ultimately decline to subscribe.

The core challenge is: given a customer's demographic profile, financial standing, and prior campaign interaction history, can we predict — before making the call — whether that customer will subscribe to a term deposit? A reliable predictive model would allow the bank to rank customers by subscription probability and selectively target only the highest-probability segments, dramatically improving the conversion rate per call and reducing overall campaign costs.

Formal Problem Statement: Given the 16 predictor features of a customer (demographics, financial attributes, and campaign history), build a binary classification model that accurately predicts whether the customer will subscribe to a term deposit (YES = 1, NO = 0), with the objective of maximizing accuracy and minimizing false negatives (missed subscribers).

3.1 Key Challenges
•	Near-Balanced Class Distribution: Unlike many classification problems, the YES/NO split is approximately 47/53%, meaning the dataset does not suffer from severe imbalance, but models must still discriminate meaningfully between classes rather than defaulting to majority voting.
•	High-Cardinality Categorical Variables: Features like job (12 categories), month (12 categories), and poutcome require careful one-hot encoding to avoid multicollinearity while preserving information.
•	pdays Column Anomaly: The pdays column records -1 for customers never previously contacted, which represented over 74% of the dataset. This near-constant value with a non-representational sentinel makes the column statistically uninformative and requires dropping.
•	Outliers in Numerical Features: Age (max: 95), balance (max: 81,204; min: -6,847), duration, campaign, and previous all contain outliers. However, these are domain-valid extremes (e.g., a 95-year-old customer is a real edge case) and were retained in the dataset.
•	Dummy Variable Trap: One-hot encoding of categorical variables must be handled with the drop-one-column strategy to prevent perfect multicollinearity, which would destabilize linear models such as Logistic Regression.
4. Project Methodology & Process
4.1 End-to-End Process Pipeline
Phase	Activity	Key Techniques/Tools
1. Data Loading	Load CSV into pandas DataFrame	pd.read_csv, df.head(), df.info()
2. EDA	Analyze all 16 features vs. deposit target	countplot, pie chart, barplot, crosstab, pairplot, heatmap
3. Data Wrangling	Outlier analysis, drop pdays, encode categoricals	boxplot, IQR analysis, pd.get_dummies, df.drop()
4. Feature Engineering	One-hot encode 9 categorical features, drop original columns	get_dummies with reference-category drop
5. Feature Scaling	Standardize numerical features	StandardScaler (fit_transform on train, transform on test)
6. Model Training	Train 6 classification models	LogisticRegression, DecisionTree, RandomForest, AdaBoost, VotingClassifier
7. Hyperparameter Tuning	Grid search over Decision Tree parameters	GridSearchCV (cv=10, max_depth, max_features)
8. Evaluation	Accuracy, confusion matrix, ROC-AUC, F1, log loss	sklearn.metrics suite
9. Business Analytics	Bin top features, analyze subscription rates per bin	pd.qcut, groupby, line plots
10. Deployment	Production deployment strategy on AWS SageMaker	SageMaker Endpoints, S3, Model Monitor

4.2 Data Preprocessing
4.2.1 Column Renaming
The target column was originally named 'y' in the raw CSV. It was renamed to 'deposit' for clarity: df.rename(columns={'y': 'deposit'}, inplace=True).

4.2.2 Data Quality Checks
•	Missing Values: df.isnull().sum() returned zero for all 17 columns — no imputation required.
•	Duplicate Rows: df.duplicated().sum() returned zero — no deduplication required.
•	Data Types: All numerical columns correctly typed as int64; all categorical columns as object dtype.

4.2.3 Outlier Analysis
Box plots were generated for all seven numerical features split by the deposit target. Statistical investigation used 1.5×IQR (Q3 threshold) as the outlier definition:
Feature	Min	Max	Q3 (75%)	1.5×Q3 Threshold	Outliers Present	Action
age	18	95	~48	~72	Yes (elderly customers)	Retained — domain valid
balance	-6,847	81,204	~1,708	~2,562	Yes (high-balance outliers)	Retained — domain valid
day	1	31	~21	~31	No	No action needed
duration	0	4,918	~317	~475	Yes (long calls)	Retained — domain valid
campaign	1	63	~3	~4.5	Yes (heavy contact)	Retained — domain valid
pdays	-1	871	—	—	74%+ are -1 sentinel	DROPPED
previous	0	275	~0	~0	Yes (heavy prior contact)	Retained — domain valid

The pdays column was dropped because 8,324 of 11,162 records (74.57%) contained the sentinel value -1 (indicating the customer had never been previously contacted). This near-constant distribution renders the feature statistically uninformative for predictive modeling.

4.2.4 Categorical Encoding — One-Hot Encoding
All 9 categorical predictor columns were encoded using pd.get_dummies() with one reference category dropped per column to avoid the dummy variable trap:
Original Column	Categories	Dummies Created	Reference Dropped
job	12 categories	11 binary columns (work_*)	work_admin.
marital	3 categories	2 binary columns (marital_status_*)	marital_status_divorced
education	4 categories	3 binary columns (quali_*)	quali_primary
default	2 categories	1 binary column (defaulter_no)	defaulter_yes
housing	2 categories	1 binary column (hloan_no)	hloan_yes
loan	2 categories	1 binary column (ploan_no)	ploan_yes
contact	3 categories	2 binary columns (contacted_*)	contacted_cellular
month	12 categories	11 binary columns (mon_*)	mon_nov
poutcome	3 categories (other→unknown)	2 binary columns (pout_*)	pout_failure

After encoding and dropping the 9 original categorical columns plus pdays, the final modeling dataset (df_1) contained the 6 original numeric features and 34 binary dummy features, totaling 40 predictors plus the binary target (depo_no).

4.3 Train-Test Split & Feature Scaling
•	Split ratio: 80% training / 20% test (train_size=0.8, random_state=101).
•	Training set: ~8,929 records  |  Test set: ~2,233 records.
•	StandardScaler applied: fit_transform() on training set, transform() only on test set to prevent data leakage.
•	Scaling ensures equal treatment of features with disparate ranges (e.g., balance in thousands vs. binary dummy features).
5. Exploratory Data Analysis (EDA)
5.1 Target Variable Distribution
The deposit target variable showed a near-balanced distribution:
Outcome	Count	Percentage	Interpretation
YES (Subscribed)	5,289	47.4%	Customer subscribed to term deposit
NO (Declined)	5,873	52.6%	Customer did not subscribe
Total	11,162	100%	Full campaign dataset

The near-balanced class distribution (47/53 split) is favorable for model training, as it minimizes class-imbalance bias and means that accuracy is a meaningful evaluation metric without requiring oversampling techniques such as SMOTE.

5.2 Feature-by-Feature Analysis vs. Deposit
5.2.1 Age vs. Deposit
A histogram of customer ages revealed that the majority of customers fall between 25 and 40 years of age. However, a comparative bar chart of age statistics between YES and NO groups showed that customers in higher age brackets have a greater propensity to subscribe to term deposits. The mean age of subscribers is higher than non-subscribers, supporting age as a meaningful predictor. Statistical summary: Mean age = 41.23, Min = 18, Max = 95.

5.2.2 Job vs. Deposit
Cross-tabulation and count plots revealed distinct subscription patterns by job type. Management-role employees and students showed the highest relative subscription rates. Blue-collar workers and service workers showed lower subscription propensity, likely reflecting different financial priorities and risk tolerances.

5.2.3 Marital Status vs. Deposit
Married customers were found to be less inclined toward term deposit subscription compared to single customers. This likely reflects competing financial commitments (mortgages, family expenses) that reduce discretionary savings capacity. Divorced customers showed intermediate subscription rates.

5.2.4 Education vs. Deposit
Customers with tertiary (university-level) education demonstrated the highest subscription rates. This is consistent with higher financial literacy among tertiary-educated customers, who may better understand and appreciate the benefits of fixed-term deposit products. Primary-education customers showed the lowest subscription rates.

5.2.5 Default Status vs. Deposit
Customers who had previously defaulted on a loan were significantly less likely to subscribe to a term deposit. This is an expected finding: past default behavior indicates financial distress and reduced capacity for additional savings commitments. The vast majority of customers in the dataset were non-defaulters.

5.2.6 Balance vs. Deposit
A comparative bar chart of account balance statistics between YES/NO groups revealed a clear positive relationship: customers who subscribed had substantially higher mean account balances than those who did not. Statistical summary: Mean balance = 1,528.53, Min = -6,847, Max = 81,204. The dataset includes customers with negative balances (overdrafts), who were less likely to subscribe.

5.2.7 Housing Loan vs. Deposit
Customers without a housing loan were more inclined to subscribe. Having a housing loan implies ongoing financial obligations, reducing the likelihood of committing additional funds to a term deposit.

5.2.8 Personal Loan vs. Deposit
Similarly, customers without a personal loan were more likely to subscribe. Both housing and personal loan flags show consistent negative relationships with term deposit subscription.

5.2.9 Contact Mode vs. Deposit
Customers contacted via cellular phone were more likely to subscribe compared to those contacted by telephone or with unknown contact type. Cellular-contacted customers may be more engaged and accessible, leading to higher conversion rates.

5.2.10 Day of Month vs. Deposit
Customers showed greater reluctance to subscribe at the beginning of the month. A bar chart of day statistics showed that call timing within the month has a measurable effect on subscription rate, with the second half of the month (post-22nd) showing better conversion.

5.2.11 Month vs. Deposit
A pronounced seasonal pattern was observed: customers were most reluctant to subscribe during summer months (May through August). November and December showed the highest relative subscription rates. This seasonal variation has direct implications for campaign scheduling.

5.2.12 Duration vs. Deposit
Call duration showed one of the strongest relationships with subscription outcome: longer calls strongly predicted subscription. Customers who eventually subscribed had substantially higher mean call durations. This makes intuitive sense — a customer engaged enough to stay on a long call is more likely to convert. Conversely, very short calls typically end in quick refusals.

5.2.13 Campaign (Number of Contacts) vs. Deposit
The number of campaign contacts showed an inverse relationship with subscription: customers contacted fewer times were more likely to subscribe. Repeatedly contacting a customer who has not yet subscribed may signal declining interest, and more contacts may indicate the customer is resistant. This has a direct implication for campaign efficiency — over-contacting customers is counterproductive.

5.2.14 Previous Contacts vs. Deposit
Fewer prior campaign contacts correlated with higher subscription rates in the current campaign. This could reflect that customers who have been heavily targeted historically develop contact fatigue.

5.2.15 Previous Outcome (poutcome) vs. Deposit
The previous campaign success rate was only 9.6% (visualized as a pie chart with 9.6% success slice). However, customers whose previous campaign outcome was 'success' showed dramatically higher subscription rates in the current campaign, making poutcome one of the most powerful predictors when available.

5.3 Correlation Analysis
A correlation heatmap (using Pearson correlation) across all numerical features was computed and visualized. Key finding: no pairs of numerical features showed strong multicollinearity (all correlation coefficients were low). This justified retaining all features rather than performing PCA or eliminating correlated pairs. The heatmap used a diverging 'twilight_shifted' colormap for clear visual separation of positive and negative correlations.

EDA Insight: The feature most strongly correlated with subscription is call duration — customers with longer calls are far more likely to subscribe. Balance and age are the next most informative numerical features. Poutcome (prior campaign success) is the most powerful categorical predictor.

6. Model Building & Numerical Results
6.1 Model Performance Summary
#	Model	Accuracy Score	Misclassified (of 2,233)	Notes
1	Logistic Regression (Optimal Threshold)	~79–80%	~447–468	Threshold optimized via ROC curve
2	GridSearchCV Decision Tree	~84–85%	~335–357	Best params: max_depth=3, max_features=None
3	Decision Tree (max_depth=5)	~84–85%	~335–357	Slight overfitting vs. GS-tuned
4	Random Forest (500 trees)	86.3%	308	BEST MODEL — Recommended
5	AdaBoost (500 estimators)	~85–86%	~312–357	base: DT max_depth=3, learning_rate=1.0
6	Voting Classifier	~85–86%	~312–357	Combines LR + GS + DT + RF

Best Model: Random Forest Classifier with 500 estimators achieved the highest accuracy of 86.3% (0.8629646215853113) and the fewest misclassifications (308 out of 2,233 test records), outperforming all other models on both metrics.

6.2 Detailed Model Analysis
6.2.1 Logistic Regression
Logistic Regression was the baseline model. After fitting the default model and generating initial predictions, the ROC curve was plotted to identify the optimal classification threshold.
•	Initial accuracy using default threshold (0.5): approximately 79–80%.
•	ROC-AUC Score: computed and plotted; curve plotted with blue line against the red dashed random classifier baseline.
•	Optimal Threshold: determined via the Youden Index (argmax of TPR − FPR from the ROC curve), which shifted the classification threshold from the default 0.5 to the optimal value, improving recall on the positive class.
•	Post-threshold Accuracy (ACS_LR): Updated accuracy and log loss computed after applying the optimal threshold.
•	Log Loss: computed using metrics.log_loss() — lower log loss indicates better probability calibration.
•	Confusion Matrix: Visualized as a coolwarm heatmap showing True Negatives (Refused → Predicted Refused), False Positives, False Negatives, and True Positives (Accepted → Predicted Accepted).
•	Limitation: The linear decision boundary of Logistic Regression is insufficient to capture the complex, non-linear interactions between features like balance, duration, and campaign count.

6.2.2 Decision Tree with GridSearchCV
GridSearchCV was applied to a DecisionTreeClassifier with 10-fold cross-validation to identify the optimal hyperparameters:
Hyperparameter	Values Searched	Optimal Value
max_depth	3, 4, 5, None	3
max_features	2, 3, 4, 5, None	None

•	Best Parameters: max_depth=3, max_features=None — a shallow tree to prevent overfitting.
•	Accuracy (ACS_GS): Approximately 84–85%, an improvement over Logistic Regression.
•	The shallow max_depth=3 constraint strongly controlled overfitting, producing good generalization to the test set.

6.2.3 Decision Tree (max_depth=5)
A standalone Decision Tree was trained with max_depth=5 (slightly deeper than the GridSearchCV optimal of 3) and max_features=None after applying StandardScaler to the features.
•	Accuracy (ACS_DT): Approximately 84–85%.
•	Feature Importance: The top features from the decision tree were dominated by duration, balance, age, and day — consistent with the Random Forest importance rankings.
•	A Graphviz visualization of the tree structure was generated, providing a fully interpretable visual explanation of the model's decision logic.

6.2.4 Random Forest Classifier — RECOMMENDED
•	Configuration: 500 decision tree estimators, n_jobs=-1 (full parallel execution), random_state=42.
•	Accuracy (ACS_RF): 86.3% — highest among all six models.
•	Misclassifications: Only 308 observations out of 2,233 test records incorrectly classified.
•	Ensemble advantage: Averaging 500 diverse decision trees dramatically reduces variance compared to a single tree, explaining why Random Forest outperforms the standalone Decision Tree.
•	Feature Importance (top 4): duration (highest importance), balance, age, day.
•	Confusion Matrix: Fewest off-diagonal values among all models, confirming superior discrimination.

6.2.5 AdaBoost Classifier
•	Configuration: 500 estimators, base estimator = DecisionTreeClassifier(criterion='gini', max_depth=3), learning_rate=1.0.
•	Accuracy (ACS_ADB): Approximately 85–86%, competitive with but slightly below Random Forest.
•	AdaBoost sequentially focuses on misclassified examples, which helps but is more sensitive to noisy data than the bagging approach of Random Forest.
•	Feature Importance: Computed and sorted; duration remained the dominant feature.

6.2.6 Voting Classifier
A hard-voting ensemble combined four base models: Logistic Regression, GridSearchCV Decision Tree, standalone Decision Tree, and Random Forest.
•	Each model was individually fitted and its accuracy printed before ensemble scoring.
•	Accuracy (ACS_V): Approximately 85–86%.
•	The Voting Classifier did not exceed Random Forest's standalone performance, suggesting the weaker models (Logistic Regression) diluted the ensemble's performance.

6.3 Final Model Comparison
Rank	Model	Accuracy	Misclassified	Verdict
1st	Random Forest (500 estimators)	86.3%	308	BEST — Recommended for production
2nd	Voting Classifier	~85–86%	~312–357	Strong but below RF alone
3rd	AdaBoost (500 estimators)	~85–86%	~312–357	Good; sensitive to noise
4th	GridSearchCV Decision Tree	~84–85%	~335–357	Good interpretability
5th	Decision Tree (max_depth=5)	~84–85%	~335–357	Slight overfit vs. GS version
6th	Logistic Regression	~79–80%	~447–468	Lowest; linear boundary insufficient

6.4 Confusion Matrix Analysis
Confusion matrices were generated for all six models and compared. The matrix structure for the binary classification problem was:
	Predicted: Refusal (0)	Predicted: Acceptance (1)
Actual: Refusal (0)	True Negatives (TN)	False Positives (FP)
Actual: Acceptance (1)	False Negatives (FN)	True Positives (TP)

Random Forest had the lowest total off-diagonal count (FP + FN = 308), confirming it produces the most accurate classification. False Negatives (missed subscribers) are particularly costly from a business perspective — they represent customers the bank failed to convert but could have, given the right targeting. Random Forest minimized this waste.

6.5 Logistic Regression — ROC Curve Analysis
•	ROC curve plotted: True Positive Rate (Sensitivity) vs. False Positive Rate (1 − Specificity).
•	AUC Score computed via roc_auc_score() — values closer to 1.0 indicate better discrimination.
•	Optimal Threshold: Identified using the Youden Index (argmax of TPR − FPR array), which shifted the decision threshold from the default 0.5 to the optimal cutoff.
•	Post-threshold update: New predictions generated using np.where(probs > optimal_threshold, 1, 0), improving accuracy from the baseline logistic score.
7. Feature Importance & Graphical Insights
7.1 Top Predictive Features (Random Forest)
Feature importances from the Random Forest model were extracted using rf_model.feature_importances_ and sorted in descending order. The four dominant features were:
Rank	Feature	Business Meaning	Direction of Effect
1st	duration	Last call duration in seconds	Higher duration → Higher subscription probability
2nd	balance	Customer bank account balance	Higher balance → Higher subscription probability
3rd	age	Customer age	Older (>53) and younger (<31) → Higher probability
4th	day	Day of the month contacted	Before 8th or after 23rd → Higher probability

A purple bar chart was generated plotting all 40 features on the x-axis (rotated 90°) and their importance scores on the y-axis, making it visually clear that duration and balance dwarf all other features in predictive power.

7.2 Business Analytics: Quantified Subscription Insights
7.2.1 Call Duration Analysis
The duration feature was binned into 10 quantile-based buckets (pd.qcut with q=10). The mean subscription rate was computed per bucket and plotted as a line chart ('Mean % subscription depending on Contact Duration').
Key Finding A1: The optimal call duration window for maximum subscription conversion is 255 to 324 seconds (approximately 4 to 6 minutes). Calls shorter than this indicate rapid disengagement; calls much longer may indicate customer confusion or reluctance. Campaign sales pitches should be designed to fit within this window.

7.2.2 Account Balance Analysis
Customer balance was segmented into 50 quantile bins. The mean subscription rate per balance bin was plotted ('Mean % subscription depending on Average Customer Balance').
Key Finding A2: Customers with account balances above approximately €935 show significantly higher subscription rates. The relationship is broadly positive — wealthier customers are more likely to invest in term deposits. The campaign focus group should prioritize customers with balances north of €935.

7.2.3 Customer Age Analysis
Age was divided into 10 quantile bins. Mean subscription rate was computed per age group and plotted ('Mean % subscription depending on Average Customer Age').
Key Finding A3: The subscription likelihood shows a U-shaped pattern with customer age. Customers younger than 31 or older than 53 are significantly more likely to subscribe. The middle age group (31–53) showed the lowest subscription rates — this cohort likely has the highest competing financial commitments (mortgages, children's education). Marketing campaigns should de-prioritize the 31–53 age bracket.

7.2.4 Contact Day of Month Analysis
The day of the month was divided into 4 quantile bins. Mean subscription rates were plotted ('Mean % subscription depending on Approach Day of the Month').
Key Finding A4: Customers contacted before the 8th or after the 23rd of the month showed higher subscription rates. The period between the 9th and 22nd showed reduced conversion — possibly because customers are mid-month occupied with regular financial commitments. Campaign call scheduling should be concentrated in the first week or final days of each month.

7.3 EDA Graphical Results Summary
Visualization	Type	Key Finding
Deposit Distribution	Pie chart	47.4% YES, 52.6% NO — near-balanced classes
Age Frequency	Histogram (15 bins)	Peak frequency: age 25–40; older customers more willing
Age vs. Deposit	Bar chart (stats)	Subscribers have higher mean age
Job vs. Deposit	Crosstab + countplot	Management & students: highest subscription rate
Marital vs. Deposit	Crosstab + countplot	Married customers least likely to subscribe
Education vs. Deposit	Crosstab + countplot	Tertiary education → highest subscription rate
Balance vs. Deposit	Bar chart (stats)	Subscribers have significantly higher mean balance
Duration vs. Deposit	Bar chart (stats)	Longer calls strongly predict subscription
Campaign vs. Deposit	Bar chart (stats)	Fewer contacts in campaign → higher subscription
Month vs. Deposit	Crosstab + countplot	Summer months (May–Aug): lowest subscription rates
poutcome Distribution	Pie chart	9.6% previous campaign success rate
Correlation Heatmap	Heatmap (twilight colormap)	No strong multicollinearity; all features retained
Pairplot	Multi-scatter (hue=deposit)	Duration and balance show best class separation
Feature Importance (RF)	Bar chart (purple)	Duration >> Balance > Age > Day dominant features

8. Deployment on AWS SageMaker
8.1 Deployment Architecture Overview
AWS SageMaker provides a comprehensive managed machine learning platform covering the full lifecycle from data preparation through model training, hyperparameter optimization, real-time inference, and production monitoring. The Random Forest model selected for the Portuguese Bank project is an ideal candidate for SageMaker deployment, offering both real-time single-customer scoring (for inbound CRM integration) and batch scoring (for weekly campaign list generation).

8.2 Full Deployment Architecture
Layer	Component	AWS Service	Purpose
Storage	Data & Model Artifacts	Amazon S3	Versioned storage for raw CSV, processed features, trained model artifacts (model.tar.gz), batch inputs/outputs
Identity	Access Management	AWS IAM	Execution roles for SageMaker with AmazonSageMakerFullAccess and AmazonS3FullAccess policies
Development	Notebook Environment	SageMaker Studio	Interactive Jupyter environment for model iteration, retraining, and experiment tracking
Training	Model Training Job	SageMaker Training Jobs	Managed training on ml.m5.xlarge with built-in sklearn container; logs to CloudWatch
Tuning	Hyperparameter Optimization	SageMaker Automatic Model Tuning	Bayesian optimization over n_estimators, max_depth, min_samples_split for Random Forest
Deployment	Real-Time Endpoint	SageMaker Endpoint	REST API endpoint on ml.m5.large for real-time per-customer churn scoring from CRM
Batch	Batch Transform Job	SageMaker Batch Transform	Weekly campaign list scoring — processes full customer base CSV overnight
Monitoring	Model Quality Monitor	SageMaker Model Monitor	Detects data drift and prediction quality degradation; alerts via SNS
Retraining	Automated Pipeline	SageMaker Pipelines	Triggered retraining when drift detected or on quarterly schedule
Visualization	Business Dashboard	Amazon QuickSight	Subscription probability scores visualized for marketing team; connected to RDS/DynamoDB

8.3 Step-by-Step Production Deployment Guide
Step 1: Environment & Permissions Setup
Create an IAM execution role with SageMaker and S3 full-access permissions. Set up versioned S3 buckets: s3://portuguese-bank-ml/data/, /models/, and /predictions/. Launch a SageMaker Studio domain for the data science team.

Step 2: Model Serialization & Packaging
Serialize the trained pipeline (StandardScaler + RandomForestClassifier) using joblib into a single pipeline object. Package the serialized model and an inference.py script (defining model_fn, input_fn, predict_fn, output_fn handlers) into a model.tar.gz archive. Upload to S3.

Step 3: Create SageMaker SKLearn Model
Instantiate a SageMaker SKLearnModel object pointing to the S3 model URI, IAM role, and the inference script. Use the managed sklearn container (sklearn version 1.0-1 or later). This ensures the same environment used in training is replicated in production.

Step 4: Deploy Real-Time Endpoint
Call model.deploy(initial_instance_count=1, instance_type='ml.m5.large', endpoint_name='portuguese-bank-deposit-endpoint'). The endpoint accepts a JSON payload of customer features and returns a JSON response with the binary prediction (0/1) and probability scores for each class. This endpoint integrates with the bank's CRM via AWS Lambda for real-time scoring of inbound customer profiles.

Step 5: Configure Batch Transform
For weekly campaign generation, configure a SageMaker Batch Transform job: point to the customer feature CSV in S3 as input, specify the endpoint model, set output location in S3, and schedule via Amazon EventBridge (weekly cron). Output predictions are ingested by the campaign management system via an AWS Glue ETL job.

Step 6: Enable Model Monitor
Create a ModelQualityMonitor schedule that captures endpoint inputs and outputs, computes prediction quality metrics against the training baseline, and publishes CloudWatch metrics. Configure SNS email notifications for drift alerts exceeding defined thresholds (e.g., accuracy drop > 3% or feature distribution shift p-value < 0.05).

Step 7: Automated Retraining Pipeline
Build a SageMaker Pipelines DAG with steps: DataPreprocessing → ModelTraining → ModelEvaluation → ConditionalRegistration (register if accuracy > 85%). Pipeline is triggered by: (a) EventBridge schedule (quarterly), (b) Model Monitor drift alert via Lambda, or (c) manual trigger from SageMaker Studio.

8.4 Inference Script Structure
Function	Role	Implementation Notes
model_fn(model_dir)	Load model from disk	joblib.load(os.path.join(model_dir, 'model.joblib'))
input_fn(request_body, content_type)	Parse input JSON/CSV to numpy array	pd.read_json or np.array parsing; apply same feature order as training
predict_fn(input_data, model)	Run prediction	model.predict(input_data) and model.predict_proba(input_data)
output_fn(prediction, accept)	Format output	JSON with {prediction, probability_no, probability_yes}

8.5 Infrastructure Cost Estimate
Component	Instance / Config	Est. Monthly Cost (USD)
Real-Time Endpoint (ml.m5.large)	1 instance, 24/7 uptime	~$100–$120
Batch Transform (ml.m5.xlarge)	Weekly, ~1–2 hours	~$5–$15
SageMaker Studio (ml.t3.medium)	Business hours only	~$15–$25
S3 Storage (data + models + logs)	< 20 GB total	~$3–$6
SageMaker Model Monitor	Per-capture charge	~$10–$20
SageMaker Pipelines	Per execution step	~$5–$10
CloudWatch Logs & Metrics	Standard monitoring	~$3–$8
Total Estimated	—	~$141–$204 / month

Note: Estimates based on AWS us-east-1 on-demand pricing. Reserved instance pricing for the endpoint can reduce cost by 30–60%. Endpoint can be scaled to zero outside business hours using SageMaker serverless inference to further reduce costs.

8.6 Business System Integration
•	CRM Integration: The SageMaker endpoint is called via AWS Lambda whenever a customer record is accessed by a customer service agent, surfacing a real-time subscription probability score in the CRM interface.
•	Campaign Management System: Weekly Batch Transform outputs feed directly into the marketing automation platform, ranking customers by subscription probability for the next campaign.
•	BI Dashboard: Prediction scores are written to Amazon DynamoDB and visualized in Amazon QuickSight, giving the marketing leadership team a real-time view of campaign targeting quality.
•	A/B Testing: SageMaker supports traffic splitting between endpoint variants — the bank can run an A/B test comparing the ML-targeted campaign against the traditional broad-reach campaign to quantify ROI improvement.
9. Conclusion
This project successfully developed and validated a machine learning solution for predicting term deposit subscriptions from the Portuguese bank's direct marketing campaign data. Starting from a raw 11,162-record CSV with 17 features, the team executed a rigorous CRISP-DM pipeline encompassing EDA, data wrangling, encoding, scaling, multi-model training and evaluation, and actionable business analytics.

9.1 Primary Conclusions
•	Random Forest Classifier is the recommended production model with an accuracy of 86.3% and only 308 misclassifications on 2,233 test records — the best performance across all six evaluated models.
•	Call duration is overwhelmingly the most important predictor of subscription. Customers engaged in longer calls (255–324 seconds / 4–6 minutes) are significantly more likely to subscribe, providing a direct and actionable metric for campaign coaching.
•	Customer financial profile (account balance) and demographics (age) are the next most important features, enabling the bank to pre-filter its customer database to the highest-probability segments before any calling is done.
•	Summer months (May–August) and mid-month timing (9th–22nd) are associated with lower subscription rates, providing clear campaign scheduling guidance.
•	The pdays column, while conceptually meaningful, was correctly identified as an uninformative field (74.57% sentinel values) and dropped — demonstrating the importance of domain-aware data cleaning over purely statistical approaches.
•	The near-balanced class distribution (47/53%) eliminated the need for oversampling, allowing standard accuracy to serve as a reliable evaluation metric.

9.2 Business Impact Summary
Recommendation	Basis	Expected Impact
Call pitch duration: 4–6 minutes (255–324 secs)	Duration bin analysis (qcut q=10)	Maximizes conversion per call; avoids over/under-selling
Target customers with balance > €935	Balance bin analysis (qcut q=50)	Higher balance → significantly higher subscription rate
Prioritize age < 31 or > 53	Age bin analysis (qcut q=10)	U-shaped age-subscription curve; avoid 31–53 cohort
Schedule calls before 8th or after 23rd of month	Day bin analysis (qcut q=4)	Monthly timing strongly affects customer receptivity
Focus on non-loan, high-education, cellular customers	EDA cross-tabulations	Multiple demographic signals compound subscription likelihood
Avoid over-contacting resistant customers	Campaign frequency vs. deposit	Diminishing returns beyond 2–3 contacts; reduces waste
Overall Conclusion: Deploying the Random Forest model via AWS SageMaker will enable No-Churn Telecom to shift from broad-reach, low-efficiency outreach to precision-targeted campaigns, directly reducing cost-per-acquisition and improving term deposit subscription rates across the customer base.

10. Further Improvements & Recommendations
10.1 Data Quality & Feature Enhancements
•	Recover pdays Information: Rather than dropping pdays entirely, create a binary flag feature has_prior_contact (1 if pdays ≠ -1, else 0) and a separate days_since_contact feature for the non-null subset. This preserves the recency signal for the 25% of customers who were previously contacted.
•	Add Customer Lifetime Value (CLV): Incorporating estimated CLV would enable value-weighted prediction — a model that prioritizes high-value customers reduces campaign cost while maximizing deposit revenue.
•	Recency-Frequency-Monetary (RFM) Features: Engineer RFM scores from balance, campaign frequency, and account tenure to create composite behavioral features with higher predictive signal.
•	Digital Channel Data: If available, integrate website browsing behavior (viewed term deposit page?), app engagement, and email open rates as additional predictors — digital signals often precede subscription intent.
•	Economic Indicators: Append macroeconomic context (ECB interest rate, Portuguese unemployment rate, EUR/USD) at the time of each campaign record. These external signals may explain seasonal patterns (summer low subscription) better than month dummies alone.

10.2 Modeling Improvements
•	Hyperparameter Tuning for Random Forest: The production model used default Random Forest parameters (n_estimators=500, default max_depth). Bayesian optimization via SageMaker Automatic Model Tuning over max_depth, min_samples_split, min_samples_leaf, and max_features could yield an additional 1–3% accuracy improvement.
•	Gradient Boosting Models: Evaluate XGBoost and LightGBM — both frequently outperform Random Forest on tabular banking data due to their handling of imbalanced feature scales and built-in regularization. LightGBM in particular offers faster training on large datasets.
•	Stacking Ensemble: A two-level stacking ensemble (Level 1: Random Forest + AdaBoost + Logistic Regression; Level 2: XGBoost meta-learner) could extract complementary signals from each base model.
•	Class-Weight Optimization: Although the class distribution is near-balanced, experimenting with class_weight='balanced' in tree-based models may improve recall for the minority (YES) class, reducing false negatives.
•	Neural Networks: For larger future datasets, a multi-layer perceptron (MLP) with dropout regularization could capture higher-order feature interactions not visible to tree-based models.

10.3 Evaluation Improvements
•	Business-Cost-Weighted Metrics: Define a custom scoring function that weights False Negatives (missed subscribers) more heavily than False Positives (unnecessary calls). This aligns model optimization with business ROI rather than symmetric accuracy.
•	Precision@K Analysis: Instead of binary classification, rank customers by predicted subscription probability and compute Precision@K (e.g., what % of the top 1,000 customers contacted actually subscribe). This directly measures campaign targeting efficiency.
•	Stratified K-Fold Cross-Validation: Replace the single train-test split with 10-fold stratified cross-validation to generate confidence intervals on model accuracy and reduce evaluation variance.
•	SHAP Values: Implement SHAP (SHapley Additive exPlanations) to generate per-customer feature attribution, enabling agents to understand why a specific customer is predicted to subscribe or decline.

10.4 Deployment & MLOps Improvements
•	SageMaker Serverless Inference: For low-traffic periods, switch the real-time endpoint to SageMaker Serverless Inference to eliminate idle compute costs while maintaining sub-second response times for burst demand.
•	Multi-Model Endpoint: Host Random Forest and AdaBoost models on a single SageMaker Multi-Model Endpoint (MME) to support A/B testing between model versions without additional infrastructure cost.
•	Feature Store: Implement Amazon SageMaker Feature Store for centralized, versioned customer feature management shared across the ML team, ensuring training-serving consistency and reducing feature engineering duplication.
•	CI/CD Pipeline: Build a full MLOps CI/CD pipeline using AWS CodePipeline + CodeBuild that automatically: (a) runs unit tests on new training code, (b) trains and evaluates a challenger model, (c) compares with the production model, and (d) promotes the challenger if it outperforms the champion.
•	Explainability Dashboard: Integrate SHAP explanations into the Amazon QuickSight dashboard so marketing managers can inspect feature contributions for individual customers or customer segments, building trust in model-driven decisions.

10.5 Business Process Integration
•	Targeted Retention Tiers: Segment customers into Low / Medium / High subscription probability tiers and assign differentiated campaign strategies — e.g., high-probability customers receive a standard pitch, medium-probability customers receive a personalized offer with an interest rate sweetener, low-probability customers are excluded.
•	Campaign Simulation Tool: Build an internal web app on top of the SageMaker endpoint where campaign managers can adjust targeting parameters (minimum balance, age range, contact month) and see real-time estimates of expected conversion count and campaign cost.
•	Feedback Loop: Capture actual campaign outcomes (subscribed / declined) and feed them back into the model retraining pipeline as labeled data, creating a continuously improving prediction system.
