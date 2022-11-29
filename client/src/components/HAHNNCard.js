import React from 'react';
import { Card, Box, CardContent, Typography } from '@mui/material';

const HAHNNCard = ({ predictions }) => {
    console.log(predictions)
    return (
    <Box style={{ marginTop: 20 }}>
        <Typography sx={{ mb: 1.5 }} style={{ fontWeight: 'bold' }}>
            Deep neural network models
        </Typography>
        <Box style={{ marginTop: 20, flexDirection: 'row', display: "flex" }}>
            <Card variant="outlined" style={{ marginRight: 25, width: 200, height: 100 }}>
                <React.Fragment>
                    <CardContent>
                        <Typography sx={{ fontSize: 14 }} color="black" gutterBottom>
                            Hierarchical attention hybrid neural network
                        </Typography>
                        <Typography sx={{ mb: 1.5 }} color={predictions?.hahnn === 1 ? "green" : "red"} style={{ fontWeight: 'bold' }}>
                            {predictions?.hahnn === 1 ? "Positive" : "Negative"}
                        </Typography>
                    </CardContent>
                </React.Fragment>
            </Card>
        </Box>
    </Box>
    )
}

export default HAHNNCard;
