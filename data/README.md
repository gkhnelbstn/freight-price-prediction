# Data: USAID Supply Chain Shipment Pricing Data (SCMS)

`scms_delivery_history.csv.gz` is the gzipped `SCMS_Delivery_History_Dataset.csv`:
10,324 line items of health-commodity shipments (HIV medicines and test kits, plus a few
malaria products) delivered to
43 countries from 2006 to 2015 by the USAID Supply Chain Management System project.

- Original source: USAID open data portal, dataset `mm7d-nzmf`
  (`https://data.usaid.gov/HIV-AIDS/Supply-Chain-Shipment-Pricing-Data/mm7d-nzmf`).
  The portal went offline in 2025.
- License: a work of the US government, in the public domain.
- Copy used here: three independent GitHub mirrors of the CSV
  (jrcinco/supply-chain-shipment-price-data, MSadriAghdam/Supply-Chain-Prediction_Neural-Network-ML,
  Mayuri-90/Shipment-Pricing-Prediction) have the same SHA-256:
  `918b992dd3e8d4b64d2a727b2c4ea607603d0c58f19484e73f7b78528c6a8673` (uncompressed).
  `freight.data.read_raw` refuses a file with any other checksum.

These are mostly air shipments of medicines, not truckload freight. Freight cost and weight
are recorded once per shipment (ASN/DN number); the other line items of that shipment point
to it with "See ASN-... (ID#:...)". Shipments with "Freight Included in Commodity Cost" or
"Invoiced Separately" have no freight number and are left out, as are 211 with no mode.
