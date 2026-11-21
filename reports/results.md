Fit on 3,076 shipments delivered before 2012, ranges calibrated on 851 from 2012, tested on 2,036 from 2013 onward.

| mode        |   shipments | model    |   mae_usd | median_ape   | within_25pct   | p10_p90_coverage   |
|:------------|------------:|:---------|----------:|:-------------|:---------------|:-------------------|
| all         |        2036 | baseline |    12,695 | 56%          | 21%            |                    |
| all         |        2036 | model    |     7,575 | 41%          | 33%            | 72%                |
| Air         |        1259 | baseline |     8,113 | 46%          | 25%            |                    |
| Air         |        1259 | model    |     6,074 | 33%          | 40%            | 75%                |
| Air Charter |         152 | baseline |    55,570 | 116%         | 18%            |                    |
| Air Charter |         152 | model    |    19,747 | 57%          | 13%            | 34%                |
| Ocean       |          52 | baseline |     6,351 | 41%          | 35%            |                    |
| Ocean       |          52 | model    |    16,143 | 157%         | 12%            | 56%                |
| Truck       |         573 | baseline |    11,965 | 72%          | 10%            |                    |
| Truck       |         573 | model    |     6,866 | 50%          | 26%            | 76%                |

Feature importance (rise in median APE when the feature is shuffled):

|                  |   median_ape_rise |
|:-----------------|------------------:|
| weight_kg        |             0.345 |
| origin           |             0.031 |
| line_quantity    |             0.03  |
| vendor_inco_term |             0.024 |
| line_value_usd   |             0.018 |
| shipment_mode    |             0.01  |
| country          |             0.01  |
| n_lines          |             0     |
| product_group    |            -0.001 |
| fulfill_via      |            -0.002 |

Drift, train vs test (PSI): verdict **retrain**

|                  |   psi |
|:-----------------|------:|
| origin           |  1.99 |
| country          |  1.53 |
| vendor_inco_term |  0.58 |
| shipment_mode    |  0.19 |
| weight_kg        |  0.07 |
| line_quantity    |  0.07 |
| line_value_usd   |  0.03 |
| product_group    |  0.02 |
| fulfill_via      |  0.01 |
| n_lines          |  0.01 |
