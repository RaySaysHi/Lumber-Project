# Lumber Delivery Supply Chain Data Dictionary

## Primary Dataset: DELIVERY TRACKDB.csv

| Variable | Type | Description |
|----------|------|-------------|
| DATE | String | Date of delivery record |
| EXPECTED TYPE A TRUCKS | Integer | Number of Type A trucks planned for the day |
| ACTUAL TYPE A TRUCKS | Integer | Number of Type A trucks actually used for the day |
| OUTSIDE EXTRA TYPE A TRUCKS | Integer | Number of additional Type A trucks rented/contracted from external providers |
| EXPECTED TYPE B FORK TRUCKS | Integer | Number of Type B fork trucks planned for the day |
| ACTUAL TYPE B TRUCKS | Integer | Number of Type B trucks actually used for the day |
| EXPECTED TYPE C TRUCKS | Integer | Number of Type C trucks planned for the day |
| ACTUAL TYPE C TRUCKS | Integer | Number of Type C trucks actually used for the day |
| LARGE DELIVERIES (LBS) | String | Total weight in pounds of large deliveries for the day |
| SMALL DELIVERIES (LBS) | String | Total weight in pounds of small deliveries for the day |
| LARGE DELIVERIES (QTY) | Integer | Number of large deliveries for the day |
| SMALL DELIVERIES (QTY) | Integer | Number of small deliveries for the day |
| RESCHEDULES (QTY) | Integer | Number of deliveries rescheduled for the day |
| RESCHEDULES (LBS) | String | Total weight in pounds of rescheduled deliveries |

## Supplementary Economic Indicators (from FRED)

| Variable | Type | Description |
|----------|------|-------------|
| WPU081 | Numeric | Producer Price Index for Lumber and Wood Products: Lumber - Measures price changes in lumber products at the wholesale level |
| GASREGW | Numeric | US Regular All Formulations Gas Price - Average price of regular gasoline across all formulations in the United States ($/gallon) |
| DFF | Numeric | Federal Funds Effective Rate - Interest rate at which banks lend reserve balances to other banks overnight (%) |

## Derived Metrics for Analysis

| Variable | Formula | Description |
|----------|---------|-------------|
| Reschedule Rate (%) | RESCHEDULES(QTY) / (LARGE DELIVERIES(QTY) + SMALL DELIVERIES(QTY)) | Percentage of total deliveries that were rescheduled |
| Weight-Adjusted Reschedule Impact | RESCHEDULES(LBS) / (LARGE DELIVERIES(LBS) + SMALL DELIVERIES(LBS)) | Proportion of total delivery weight that was rescheduled |
| Truck Utilization Efficiency | (ACTUAL TYPE A + ACTUAL TYPE B + ACTUAL TYPE C) / (EXPECTED TYPE A + EXPECTED TYPE B + EXPECTED TYPE C) | Ratio of actual to expected trucks, measuring planning accuracy |
| Delivery Fulfillment Rate | 1 - Reschedule Rate | Percentage of deliveries completed as scheduled |
| Type A Truck Accuracy | ACTUAL TYPE A TRUCKS / EXPECTED TYPE A TRUCKS | Ratio measuring planning accuracy for Type A trucks |
| Type B Truck Accuracy | ACTUAL TYPE B TRUCKS / EXPECTED TYPE B FORK TRUCKS | Ratio measuring planning accuracy for Type B trucks |
| Type C Truck Accuracy | ACTUAL TYPE C TRUCKS / EXPECTED TYPE C TRUCKS | Ratio measuring planning accuracy for Type C trucks |
| Average Delivery Size | (LARGE DELIVERIES(LBS) + SMALL DELIVERIES(LBS)) / (LARGE DELIVERIES(QTY) + SMALL DELIVERIES(QTY)) | Average weight per delivery in pounds |

## Seasonal Components (from Additive Decomposition)

| Component | Description |
|-----------|-------------|
| Trend | Long-term progression of delivery metrics after removing seasonal and irregular components |
| Seasonal | Recurring patterns at fixed intervals (e.g., weekly, monthly, quarterly) |
| Residual | Irregular component after removing trend and seasonal effects |
