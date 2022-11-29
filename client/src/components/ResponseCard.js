import React from 'react';
import { Card, Box, CardContent, Typography } from '@mui/material';

const model_name = {
    prediction_lr_bow: 'Logistic regression - Bag of Words',
    prediction_lr_tfidf: 'Logistic regression - TFIDF',
    prediction_mnb_bow: 'Multinomial Naive Bayes - Bag of Words',
    prediction_mnb_tfidf: 'Multinomial Naive Bayes - TFIDF',
    prediction_svc_bow: 'Support Vector Machine - Bag of Words',
    prediction_svc_tfidf: 'Support Vector Machine - TFIDF'
}

const ResponseCard = ({ predictions }) => {
    return (
    <Box style={{ marginTop: 20 }}>
        <Typography sx={{ mb: 1.5 }} style={{ fontWeight: 'bold' }}>
            Linear model predictions
        </Typography>
        <Box style={{ marginTop: 20, flexDirection: 'row', display: "flex" }}>
            {Object.keys(predictions).map((prediction) => {
                const key = prediction
                const value = predictions[prediction]
                return (
                <Card variant="outlined" style={{ marginRight: 25, width: 200, height: 100 }}>
                    <React.Fragment>
                        <CardContent>
                            <Typography sx={{ fontSize: 14 }} color="black" gutterBottom>
                                {model_name[key]}
                            </Typography>
                            <Typography sx={{ mb: 1.5 }} color={value === 1 ? "green" : "red"} style={{ fontWeight: 'bold' }}>
                                {value === 1 ? 'Positive' : 'Negative'}
                            </Typography>
                        </CardContent>
                    </React.Fragment>
                </Card>
            )
            })}
        </Box>
    </Box>
    )
}

export default ResponseCard;
