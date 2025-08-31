import great_expectations as ge
import pandas as pd
from typing import Tuple, List

df = pd.read_csv(
    "/Users/abhishekseth/Desktop/Development/Telco_ML_E2E/Telco-Customer-Churn-ML/data/raw/Telco-Customer-Churn.csv"
)


def validate_telco_data(df) -> Tuple[bool, List[str]]:
    """
    Comprehensive data validation for Telco Customer Churn dataset using Great Expectations.

    This function implements critical data quality checks that must pass before model training.
    It validates data integrity, business logic constraints, and statistical properties
    that the ML model expects.

    """
    print("🔍 Starting data validation with Great Expectations...")

    # Convert pandas DataFrame to Great Expectations Dataset
    # ge_df = ge.dataset.PandasDataset(df)

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    context = ge.get_context()

    # 4- Connect to data and create a Batch.
    # Define a Data Source, Data Asset, Batch Definition, and Batch. The Pandas DataFrame is provided to the Batch Definition at runtime to create the Batch.
    data_source = context.data_sources.add_pandas("pandas")
    data_asset = data_source.add_dataframe_asset(name="telco_churn_data")

    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        "telco_churn_data_whole_dataframe"
    )
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    suite = context.suites.add(ge.ExpectationSuite(name="telco_churn_data_suite"))
    validation_definition = ge.ValidationDefinition(
        data=batch_definition, suite=suite, name="telco_churn_data_validation"
    )

    # === SCHEMA VALIDATION - ESSENTIAL COLUMNS ===
    print("   📋 Validating schema and required columns...")

    # # Customer identifier must exist (required for business operations)
    # ge_df.expect_column_to_exist("customerID")
    # ge_df.expect_column_values_to_not_be_null("customerID")

    expectation1 = ge.expectations.ExpectColumnToExist(column="customerID")
    expectation2 = ge.expectations.ExpectColumnValuesToNotBeNull(column="customerID")

    # # Core demographic features
    # ge_df.expect_column_to_exist("gender")
    # ge_df.expect_column_to_exist("Partner")
    # ge_df.expect_column_to_exist("Dependents")

    expectation3 = ge.expectations.ExpectColumnToExist(column="gender")
    expectation4 = ge.expectations.ExpectColumnToExist(column="Partner")
    expectation5 = ge.expectations.ExpectColumnToExist(column="Dependents")

    # # Service features (critical for churn analysis)
    # ge_df.expect_column_to_exist("PhoneService")
    # ge_df.expect_column_to_exist("InternetService")
    # ge_df.expect_column_to_exist("Contract")

    expectation6 = ge.expectations.ExpectColumnToExist(column="PhoneService")
    expectation7 = ge.expectations.ExpectColumnToExist(column="InternetService")
    expectation8 = ge.expectations.ExpectColumnToExist(column="Contract")

    # # Financial features (key churn predictors)
    # ge_df.expect_column_to_exist("tenure")
    # ge_df.expect_column_to_exist("MonthlyCharges")
    # ge_df.expect_column_to_exist("TotalCharges")

    expectation9 = ge.expectations.ExpectColumnToExist(column="tenure")
    expectation10 = ge.expectations.ExpectColumnToExist(column="MonthlyCharges")
    expectation11 = ge.expectations.ExpectColumnToExist(column="TotalCharges")

    # # === BUSINESS LOGIC VALIDATION ===
    print("   💼 Validating business logic constraints...")

    # # Gender must be one of expected values (data integrity)
    # ge_df.expect_column_values_to_be_in_set("gender", ["Male", "Female"])

    expectation12 = ge.expectations.ExpectColumnDistinctValuesToBeInSet(
        column="gender", value_set=["Male", "Female"]
    )

    # # Yes/No fields must have valid values
    # ge_df.expect_column_values_to_be_in_set("Partner", ["Yes", "No"])
    # ge_df.expect_column_values_to_be_in_set("Dependents", ["Yes", "No"])
    # ge_df.expect_column_values_to_be_in_set("PhoneService", ["Yes", "No"])

    expectation13 = ge.expectations.ExpectColumnDistinctValuesToBeInSet(
        column="Partner", value_set=["Yes", "No"]
    )
    expectation14 = ge.expectations.ExpectColumnDistinctValuesToBeInSet(
        column="Dependents", value_set=["Yes", "No"]
    )
    expectation15 = ge.expectations.ExpectColumnDistinctValuesToBeInSet(
        column="PhoneService", value_set=["Yes", "No"]
    )

    # # Contract types must be valid (business constraint)
    # ge_df.expect_column_values_to_be_in_set(
    #     "Contract", ["Month-to-month", "One year", "Two year"]
    # )

    expectation16 = ge.expectations.ExpectColumnDistinctValuesToBeInSet(
        column="Contract", value_set=["Month-to-month", "One year", "Two year"]
    )

    # # Internet service types (business constraint)
    # ge_df.expect_column_values_to_be_in_set(
    #     "InternetService", ["DSL", "Fiber optic", "No"]
    # )

    expectation17 = ge.expectations.ExpectColumnDistinctValuesToBeInSet(
        column="InternetService", value_set=["DSL", "Fiber optic", "No"]
    )

    # # === NUMERIC RANGE VALIDATION ===
    print("   📊 Validating numeric ranges and business constraints...")

    # # Tenure must be non-negative (business logic - can't have negative tenure)
    # ge_df.expect_column_values_to_be_between("tenure", min_value=0)

    expectation18 = ge.expectations.ExpectColumnValuesToBeBetween(
        column="tenure", min_value=0
    )

    # # Monthly charges must be positive (business logic - no free service)
    # ge_df.expect_column_values_to_be_between("MonthlyCharges", min_value=0)

    expectation19 = ge.expectations.ExpectColumnValuesToBeBetween(
        column="MonthlyCharges", min_value=0
    )

    # # Total charges should be non-negative (business logic)
    # ge_df.expect_column_values_to_be_between("TotalCharges", min_value=0)

    expectation20 = ge.expectations.ExpectColumnValuesToBeBetween(
        column="TotalCharges", min_value=0
    )

    # # === STATISTICAL VALIDATION ===
    print("   📈 Validating statistical properties...")

    # # Tenure should be reasonable (max ~10 years = 120 months for telecom)
    # ge_df.expect_column_values_to_be_between("tenure", min_value=0, max_value=120)

    expectation21 = ge.expectations.ExpectColumnValuesToBeBetween(
        column="tenure", min_value=0, max_value=120
    )

    # # Monthly charges should be within reasonable business range
    # ge_df.expect_column_values_to_be_between(
    #     "MonthlyCharges", min_value=0, max_value=200
    # )

    expectation22 = ge.expectations.ExpectColumnValuesToBeBetween(
        column="MonthlyCharges", min_value=0, max_value=200
    )

    # # No missing values in critical numeric features
    # ge_df.expect_column_values_to_not_be_null("tenure")
    # ge_df.expect_column_values_to_not_be_null("MonthlyCharges")

    expectation23 = ge.expectations.ExpectColumnValuesToNotBeNull(column="tenure")
    expectation24 = ge.expectations.ExpectColumnValuesToNotBeNull(
        column="MonthlyCharges"
    )

    # # === DATA CONSISTENCY CHECKS ===
    print("   🔗 Validating data consistency...")

    # # Total charges should generally be >= Monthly charges (except for very new customers)
    # # This is a business logic check to catch data entry errors
    # ge_df.expect_column_pair_values_A_to_be_greater_than_B(
    #     column_A="TotalCharges",
    #     column_B="MonthlyCharges",
    #     or_equal=True,
    #     mostly=0.95,  # Allow 5% exceptions for edge cases
    # )

    expectation25 = ge.expectations.ExpectColumnPairValuesAToBeGreaterThanB(
        column_A="TotalCharges",
        column_B="MonthlyCharges",
        or_equal=True,
        mostly=0.95,  # Allow 5% exceptions for edge cases
    )

    ## Adding the expectations to the suite
    suite.add_expectation(expectation1)
    suite.add_expectation(expectation2)
    suite.add_expectation(expectation3)
    suite.add_expectation(expectation4)
    suite.add_expectation(expectation5)
    suite.add_expectation(expectation6)
    suite.add_expectation(expectation7)
    suite.add_expectation(expectation8)
    suite.add_expectation(expectation9)
    suite.add_expectation(expectation10)
    suite.add_expectation(expectation11)
    suite.add_expectation(expectation12)
    suite.add_expectation(expectation13)
    suite.add_expectation(expectation14)
    suite.add_expectation(expectation15)
    suite.add_expectation(expectation16)
    suite.add_expectation(expectation17)
    suite.add_expectation(expectation18)
    suite.add_expectation(expectation19)
    suite.add_expectation(expectation20)
    suite.add_expectation(expectation21)
    suite.add_expectation(expectation22)
    suite.add_expectation(expectation23)
    suite.add_expectation(expectation24)
    suite.add_expectation(expectation25)

    # # === RUN VALIDATION SUITE ===
    print("   ⚙️  Running complete validation suite...")
    # results = ge_df.validate()

    results = batch.validate(suite)

    # # === PROCESS RESULTS ===
    # # Extract failed expectations for detailed error reporting
    # failed_expectations = []
    # for r in results["results"]:
    #     if not r["success"]:
    #         expectation_type = r["expectation_config"]["expectation_type"]
    #         failed_expectations.append(expectation_type)

    failed_expectations = []
    for r in results["results"][0:]:
        if not r["success"]:
            failed_expectations.append(r["expectation_config"]["type"])
            # failed_expectations.append(
            #     r["expectation_config"]["type"]
            #     + ": "
            #     + str(r["expectation_config"]["kwargs"]["column"])
            # )

    # # Print validation summary
    total_checks = len(results["results"])
    passed_checks = sum(1 for r in results["results"][0:] if r["success"])
    failed_checks = total_checks - passed_checks

    if results["success"]:
        print(
            f"✅ Data validation PASSED: {passed_checks}/{total_checks} checks successful"
        )
    else:
        print(
            f"❌ Data validation FAILED: {failed_checks}/{total_checks} checks failed"
        )
        print(f"   Failed expectations: {failed_expectations}")

    return results["success"], failed_expectations


#########################################################################################################################
# import great_expectations as ge
# from typing import Tuple, List


# def validate_telco_data(df) -> Tuple[bool, List[str]]:
#     """
#     Comprehensive data validation for Telco Customer Churn dataset using Great Expectations.

#     This function implements critical data quality checks that must pass before model training.
#     It validates data integrity, business logic constraints, and statistical properties
#     that the ML model expects.

#     """
#     print("🔍 Starting data validation with Great Expectations...")

#     # Convert pandas DataFrame to Great Expectations Dataset
#     # ge_df = ge.dataset.PandasDataset(df)
#     ge_df = ge.from_pandas(df)

#     # === SCHEMA VALIDATION - ESSENTIAL COLUMNS ===
#     print("   📋 Validating schema and required columns...")

#     # Customer identifier must exist (required for business operations)
#     ge_df.expect_column_to_exist("customerID")
#     ge_df.expect_column_values_to_not_be_null("customerID")

#     # Core demographic features
#     ge_df.expect_column_to_exist("gender")
#     ge_df.expect_column_to_exist("Partner")
#     ge_df.expect_column_to_exist("Dependents")

#     # Service features (critical for churn analysis)
#     ge_df.expect_column_to_exist("PhoneService")
#     ge_df.expect_column_to_exist("InternetService")
#     ge_df.expect_column_to_exist("Contract")

#     # Financial features (key churn predictors)
#     ge_df.expect_column_to_exist("tenure")
#     ge_df.expect_column_to_exist("MonthlyCharges")
#     ge_df.expect_column_to_exist("TotalCharges")

#     # === BUSINESS LOGIC VALIDATION ===
#     print("   💼 Validating business logic constraints...")

#     # Gender must be one of expected values (data integrity)
#     ge_df.expect_column_values_to_be_in_set("gender", ["Male", "Female"])

#     # Yes/No fields must have valid values
#     ge_df.expect_column_values_to_be_in_set("Partner", ["Yes", "No"])
#     ge_df.expect_column_values_to_be_in_set("Dependents", ["Yes", "No"])
#     ge_df.expect_column_values_to_be_in_set("PhoneService", ["Yes", "No"])

#     # Contract types must be valid (business constraint)
#     ge_df.expect_column_values_to_be_in_set(
#         "Contract", ["Month-to-month", "One year", "Two year"]
#     )

#     # Internet service types (business constraint)
#     ge_df.expect_column_values_to_be_in_set(
#         "InternetService", ["DSL", "Fiber optic", "No"]
#     )

#     # === NUMERIC RANGE VALIDATION ===
#     print("   📊 Validating numeric ranges and business constraints...")

#     # Tenure must be non-negative (business logic - can't have negative tenure)
#     ge_df.expect_column_values_to_be_between("tenure", min_value=0)

#     # Monthly charges must be positive (business logic - no free service)
#     ge_df.expect_column_values_to_be_between("MonthlyCharges", min_value=0)

#     # Total charges should be non-negative (business logic)
#     ge_df.expect_column_values_to_be_between("TotalCharges", min_value=0)

#     # === STATISTICAL VALIDATION ===
#     print("   📈 Validating statistical properties...")

#     # Tenure should be reasonable (max ~10 years = 120 months for telecom)
#     ge_df.expect_column_values_to_be_between("tenure", min_value=0, max_value=120)

#     # Monthly charges should be within reasonable business range
#     ge_df.expect_column_values_to_be_between(
#         "MonthlyCharges", min_value=0, max_value=200
#     )

#     # No missing values in critical numeric features
#     ge_df.expect_column_values_to_not_be_null("tenure")
#     ge_df.expect_column_values_to_not_be_null("MonthlyCharges")

#     # === DATA CONSISTENCY CHECKS ===
#     print("   🔗 Validating data consistency...")

#     # Total charges should generally be >= Monthly charges (except for very new customers)
#     # This is a business logic check to catch data entry errors
#     ge_df.expect_column_pair_values_A_to_be_greater_than_B(
#         column_A="TotalCharges",
#         column_B="MonthlyCharges",
#         or_equal=True,
#         mostly=0.95,  # Allow 5% exceptions for edge cases
#     )

#     # === RUN VALIDATION SUITE ===
#     print("   ⚙️  Running complete validation suite...")
#     results = ge_df.validate()

#     # === PROCESS RESULTS ===
#     # Extract failed expectations for detailed error reporting
#     failed_expectations = []
#     for r in results["results"]:
#         if not r["success"]:
#             expectation_type = r["expectation_config"]["expectation_type"]
#             failed_expectations.append(expectation_type)

#     # Print validation summary
#     total_checks = len(results["results"])
#     passed_checks = sum(1 for r in results["results"] if r["success"])
#     failed_checks = total_checks - passed_checks

#     if results["success"]:
#         print(
#             f"✅ Data validation PASSED: {passed_checks}/{total_checks} checks successful"
#         )
#     else:
#         print(
#             f"❌ Data validation FAILED: {failed_checks}/{total_checks} checks failed"
#         )
#         print(f"   Failed expectations: {failed_expectations}")

#     return results["success"], failed_expectations
