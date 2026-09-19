# Research Basis

The model combines ideas from drug-shortage prediction, pharmaceutical demand forecasting, spatiotemporal disease forecasting, graph neural networks, and news-based outbreak intelligence.

## Drug Shortage Prediction

Pall et al. showed that machine learning over pharmacy sales and shortage labels can predict many impactful shortages one month ahead without direct inventory data. This supports the project assumption that demand and external signals can improve shortage-risk forecasts even when shelf inventory is unavailable.

- Pall, Gauthier et al., "Predicting drug shortages using pharmacy data and machine learning", PLOS Digital Health / PubMed Central: `https://pmc.ncbi.nlm.nih.gov/articles/PMC10009839/`

Recent shortage prediction work also emphasizes occurrence, duration, shortage cause, and historical incidence frequency as useful targets and predictors. The architecture therefore separates supply disruption risk, shortage duration/impact, and local demand exposure instead of treating shortage as one binary label.

- Frontiers in Pharmacology, "Drug shortage in South Korea: machine learning-based prediction...", `https://www.frontiersin.org/journals/pharmacology/articles/10.3389/fphar.2025.1608843/full`

## Pharmaceutical Demand Forecasting

Pharmaceutical demand forecasting literature supports ML and deep learning for nonlinear, seasonal, and cross-product effects. Knowledge graph and GCN/LSTM-style methods are especially relevant when drug substitutions, comorbidities, and disease seasonality create dependencies across products.

- Scientific Reports article on knowledge graph-enhanced deep learning for pharmaceutical demand forecasting: `https://www.nature.com/articles/s41598-026-35113-4`
- "Pharmaceutical Demand Forecasting via GCN-LSTM: A Knowledge..." preprint: `https://www.preprints.org/manuscript/202510.1546`

## Spatiotemporal Disease and Graph Forecasting

Disease spread and health demand are spatially coupled. Graph neural networks and hybrid disease models have been used to improve regional forecasting by learning spatial/temporal dependencies.

- Fritz, Dorigatti, Ruegamer, "Combining graph neural networks and spatio-temporal disease models to improve the prediction of weekly COVID-19 cases in Germany", Scientific Reports: `https://www.nature.com/articles/s41598-022-07757-5`
- Kapoor et al., "Examining COVID-19 Forecasting using Spatio-Temporal Graph Neural Networks", arXiv: `https://arxiv.org/abs/2007.03113`
- "A Review of Graph Neural Networks in Epidemic Modeling", arXiv: `https://arxiv.org/html/2403.19852v2`
- BMC Public Health GNN influenza prediction article: `https://link.springer.com/article/10.1186/s12889-025-21618-6`

These papers justify using county/region graphs, distance attenuation, and learned embeddings to project news/disease events into Arkansas forecast geographies.

## News and Language-Model Event Extraction

Outbreak intelligence systems use news and text mining as early warning complements to traditional surveillance. This supports a text extraction layer that converts unstructured news into structured location/disease/drug/supplier/event features.

- "AI-driven epidemic intelligence: the future of outbreak detection and..." PubMed Central: `https://pmc.ncbi.nlm.nih.gov/articles/PMC12343573/`
- "BioCaster in 2021: automatic disease outbreaks detection from online news", PDF: `https://eprints.gla.ac.uk/274437/1/274437.pdf`
- "Health Sentinel: An AI Pipeline For Real-time Disease Outbreak..." ACL Anthology PDF: `https://aclanthology.org/2025.nlp4pi-1.3.pdf`
- "Attention-Driven Deep Learning for News-Based Prediction of Disease Outbreaks", MDPI Big Data and Cognitive Computing: `https://www.mdpi.com/2504-2289/9/11/291`

The important modeling constraint is that text models should produce evidence features, not final forecasts. Final forecasts should be learned from historical relationships among text-derived events, disease surveillance, weather, demand, and shortage labels.

## Practical Modeling Guidance

1. Keep a simple baseline for every target.
2. Use time-based validation only.
3. Evaluate rare shortage events with precision-recall, calibration, and top-k recall, not accuracy alone.
4. Model Arkansas shortage impact as national disruption risk multiplied or learned against Arkansas drug exposure.
5. Use multi-task learning so sparse shortage labels can benefit from richer demand, disease, weather, and recall signals.
6. Maintain interpretability with driver summaries from regression coefficients, permutation importance, ablation, or SHAP-like local contributions when available.
