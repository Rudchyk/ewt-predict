import logo from '/energy-web-token-ewt-logo.svg';
import { useState, useEffect, type ChangeEvent } from 'react';
import {
  Typography,
  Container,
  ThemeProvider,
  createTheme,
  CssBaseline,
  Avatar,
  Stack,
  TextField,
  Fab,
  Alert,
  Chip,
} from '@mui/material';
import axios, { AxiosError } from 'axios';
import FileUploadIcon from '@mui/icons-material/FileUpload';
import CircularProgress from '@mui/material/CircularProgress';
import type { PredictResponse } from './Interfaces';
import { Chart } from './Chart';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import { format } from 'date-fns';
import CalendarMonthIcon from '@mui/icons-material/CalendarMonth';

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#A566FF',
    },
  },
});

function App() {
  const [intoFuture, setIntoFuture] = useState<number>(30);
  // const [windowSize, setWindowSize] = useState<number>(30);
  const [response, setResponse] = useState<PredictResponse | null>(null);
  const [message, setMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const getPredict = async () => {
    setIsLoading(true);
    setMessage('');
    setResponse(null);
    try {
      const { data } = await axios.get('/api/predict', {
        params: {
          into_future: intoFuture,
          // window_size: windowSize,
        },
      });
      setResponse(data);
    } catch (error) {
      const err = error as AxiosError<{ error: string }>;
      if (err?.response?.data) {
        setMessage(err.response.data.error);
      } else {
        setMessage(err.message);
      }
    } finally {
      setIsLoading(false);
    }
  };
  const handleIntoFutureOnChange = (e: ChangeEvent<HTMLInputElement>) => {
    setIntoFuture(Number(e.target.value));
  };
  // const handleWindowSizeOnChange = (e: ChangeEvent<HTMLInputElement>) => {
  //   setWindowSize(Number(e.target.value));
  // };

  useEffect(() => {
    getPredict();

    return () => {
      setResponse(null);
    };
  }, []);

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Container maxWidth="lg" sx={{ py: 2 }}>
        <Stack alignItems="center" spacing={2}>
          <Avatar alt="EWT logo" src={logo} sx={{ width: 56, height: 56 }} />
          <Typography variant="h4" textAlign="center">
            Передбачення Energy Web Token (не івестиційна ідея!)
          </Typography>
          <Stack direction="row" spacing={2}>
            <TextField
              value={intoFuture}
              label="Оберіть кількість днів для передбачення"
              variant="outlined"
              type="number"
              onChange={handleIntoFutureOnChange}
            />
            {/* <TextField
              value={windowSize}
              label="Оберіть кількість днів у вікні даних"
              variant="outlined"
              type="number"
              onChange={handleWindowSizeOnChange}
            /> */}
            <Fab color="primary" aria-label="get" onClick={getPredict}>
              <FileUploadIcon />
            </Fab>
          </Stack>
          {isLoading ? (
            <CircularProgress />
          ) : (
            <>
              {!!message && <Alert severity="error">{message}</Alert>}
              {!!response && (
                <Stack spacing={2} width="100%">
                  <Stack
                    direction={{ xs: 'column', md: 'row' }}
                    spacing={2}
                    alignItems="center"
                    justifyContent="center"
                  >
                    <Typography variant="h5">Найкраща ціна буде:</Typography>
                    <Typography variant="h3">
                      {response?.max_price?.toFixed(4)} EWT/USD
                    </Typography>
                    <Chip
                      icon={<CalendarMonthIcon />}
                      label={format(response?.max_date, 'dd-MM-yyyy')}
                      variant="outlined"
                    />
                    <Fab
                      color="secondary"
                      size="small"
                      aria-label="get"
                      target="_blank"
                      href={response?.plot || ''}
                    >
                      <ShowChartIcon />
                    </Fab>
                  </Stack>
                  <Chart
                    actual_prices={response?.actual_prices}
                    predicted_prices={response?.predicted_prices}
                  />
                </Stack>
              )}
            </>
          )}
        </Stack>
      </Container>
    </ThemeProvider>
  );
}

export default App;
