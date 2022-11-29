import { Grid, FormControl, Select, MenuItem, Typography, CircularProgress } from '@mui/material';
import React, { useEffect, useState } from 'react';
import HAHNNCard from './HAHNNCard';
import ResponseCard from './ResponseCard';

const sentences = [
    'Lame, lame, lame!!! A 90-minute cringe-fest thats 89 minutes too long. A setting ripe with atmosphere and possibility (an abandoned convent) is squandered by a stinker of a script filled with clunky, witless dialogue thats straining oh-so-hard to be hip. Mostly its just embarrassing, and the attempts at gonzo horror fall flat (a sample of this movies dialogue: after demonstrating her artillery, fast dolly shot to a closeup of Barbeaus vigilante charactershe: any questions? hyuck hyuck hyuck). Bad acting, idiotic, homophobic jokes and judging from the creature effects, it looks like the directors watched The Evil Dead way too many times. <br /><br />I owe my friends big time for renting this turkey and subjecting them to ninety wasted minutes theyll never get back. What a turd.',
    'Actor turned director Bill Paxton follows up his promising debut, the Gothic-horror ""Frailty"", with this family friendly sports drama about the 1913 U.S. Open where a young American caddy rises from his humble background to play against his Bristish idol in what was dubbed as ""The Greatest Game Ever Played."" Im no fan of golf, and these scrappy underdog sports flicks are a dime a dozen most recently done to grand effect with ""Miracle"" and ""Cinderella Man"", but some how this film was enthralling all the same.The film starts with some creative opening credits imagine a Disneyfied version of the animated opening credits of HBOs ""Carnivale"" and ""Rome"", but lumbers along slowly for its first by-the-numbers hour. Once the action moves to the U.S. Open things pick up very well. Paxton does a nice job and shows a knack for effective directorial flourishes I loved the rain-soaked montage of the action on day two of the open that propel the plot further or add some unexpected psychological depth to the proceedings. Theres some compelling character development when the British Harry Vardon is haunted by images of the aristocrats in black suits and top hats who destroyed his family cottage as a child to make way for a golf course. He also does a good job of visually depicting what goes on in the players heads under pressure. Golf, a painfully boring sport, is brought vividly alive here. Credit should also be given the set designers and costume department for creating an engaging period-piece atmosphere of London and Boston at the beginning of the twentieth century.You know how this is going to end not only because its based on a true story but also because films in this genre follow the same template over and over, but Paxton puts on a better than average show and perhaps indicates more talent behind the camera than he ever had in front of it. Despite the formulaic nature, this is a nice and easy film to root for that deserves to find an audience.',
    'William Hurt may not be an American matinee idol anymore, but he still has pretty good taste in B-movie projects. Here, he plays a specialist in hazardous waste clean-ups with a tragic past tracking down a perennial loser on the run --played by former pretty-boy Weller-- who has been contaminated with a deadly poison. Current pretty-boy Hardy Kruger Jr --possibly more handsome than his dad-- is featured as Wellers arrogant boss in a horrifying sequence at a chemical production plant which gets the story moving. Natasha McElhone is a slightly wacky government agent looking into the incident who provides inevitable & high-cheekboned love interest for hero Hurt. Michael Brandon pops up to play a slimy take-no-prisoners type whose comeuppance you cant wait for. The Coca-Cola company wins the Product Placement award for 2000 as the soft drink is featured throughout the production, shot lovingly on location in a wintery picture-postcard Hungary.'
]

const SentenceSelection = () => {
    const [review, setReviews] = useState('')
    const [loading, setLoading] = useState(false);
    const [hahnn_loading, setHahnnLoading] = useState(false);
    const [predictions, setPredictions] = useState({ hahnn: 1 });
    const [predictions_hahnn, setPredictions_hahnn] = useState({ hahnn: 1 });

    const handleChange = (event) => {
        setLoading(true)
        setHahnnLoading(true)
        setReviews(event.target.value)
    }

    useEffect(() => {
        if (review !== '') {
            fetch('http://localhost:5000/', {
                method: "POST",
                body: JSON.stringify({ text: review }),
            }).then((response) => {
                if (response.ok) {
                    return response.json()
                }
            }).then((res) => {
                setLoading(false)
                setPredictions(res)
            }).catch((err) => {
                setLoading(false)
                console.log(err)
            });
            
            fetch('http://192.168.0.55:5000/hahnn', {
                method: "POST",
                body: JSON.stringify({ text: review }),
            }).then((response) => {
                if (response.ok) {
                    return response.json()
                }
            }).then((res) => {
                setPredictions_hahnn({ hahnn: res === 'neg' ? 0 : 1 })
                setHahnnLoading(false)
            }).catch((err) => {
                setHahnnLoading(false)
                console.log(err)
            })
        }
    }, [review])

    return (
        <Grid container mt={4}>
            <Typography variant='body1'>Select input text to run Sentiment Analysis</Typography>
            <FormControl fullWidth variant='outlined' style={{ paddingTop: 25 }}>
                <Typography variant='body1'>Reviews</Typography>
                <Select
                    value={review}
                    onChange={handleChange}
                >
                    {sentences.map((sentence) => (
                        <MenuItem value={sentence}>{sentence}</MenuItem>
                    ))}
                </Select>
            </FormControl>
            {(loading || hahnn_loading) ? <CircularProgress style={{ marginTop: 35 }} /> : (review !== '' ? <ResponseCard predictions={predictions} /> : null)}
            {!loading && !hahnn_loading && review !== '' && <HAHNNCard predictions={predictions_hahnn} />}
        </Grid>
    )
}

export default SentenceSelection;
