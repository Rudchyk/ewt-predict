export interface ChartItem {
  date: string;
  price: number;
}

export type Chart = ChartItem[];

export interface PredictResponse {
  max_date: string;
  max_price: number;
  actual_prices: Chart;
  predicted_prices: Chart;
  plot: string;
}
