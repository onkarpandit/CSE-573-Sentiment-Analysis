import React from 'react';
import { Grid, Typography } from '@mui/material';
import SentenceSelection from './components/SentenceSelection';

const styles = {
  container: {
    height: '100%',
    paddingLeft: 50,
    paddingRight: 50,
    paddingTop: 30,
    paddingBottom: 50
  }
}

const App = () => {
  return (
    <Grid container justifyContent='space-between' direction='column' style={styles.container}>
      <Typography variant='h4'>Sentiment analysis</Typography>
      <SentenceSelection />
    </Grid>
  )
}

export default App;
