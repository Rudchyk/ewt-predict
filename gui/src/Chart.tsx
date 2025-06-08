import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import type { Chart as IChart } from './Interfaces';
import type { FC } from 'react';
import { Box } from '@mui/material';
import { format } from 'date-fns';

interface ChartProps {
  actual_prices?: IChart;
  predicted_prices?: IChart;
}

interface DataItem {
  date: string;
  price?: number;
  predicted_price?: number;
}

type Data = DataItem[];

export const Chart: FC<ChartProps> = ({
  actual_prices = [],
  predicted_prices = [],
}) => {
  console.log(predicted_prices);
  const normalizedPredictedPrices = predicted_prices.map(({ date, price }) => ({
    date: format(date, 'dd-MM-yyyy'),
    predicted_price: price,
  }));
  const data: Data = [...actual_prices, ...normalizedPredictedPrices];
  return (
    <Box width="100%" height={500}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" />
          <YAxis />
          <Tooltip
            labelStyle={{
              color: 'black',
            }}
            formatter={(value) => [`${value} EWT/USD`, 'Price']}
          />
          <Legend />
          <Line type="monotone" dataKey="price" dot={false} />
          <Line
            type="monotone"
            dataKey="predicted_price"
            dot={false}
            stroke="red"
          />
        </LineChart>
      </ResponsiveContainer>
    </Box>
  );
};

export default Chart;
