// eslint-disable-next-line no-unused-vars
import { createTheme } from "@mui/material/styles";

export const primary = {
  main: "#2352D0",
  contrastText: "#fff",
};

export const secondary = {
  main: "#707070",
  contrastText: "#fff",
};

const theme = createTheme({
  palette: {
    primary,
    secondary,
  },

  typography: {
    fontFamily: "Arial",
    button: {
      textTransform: "none",
    },
  },
});

window.theme = theme;

export default theme;
