// import axios from axios

async function useGetPredictions(text) {
    // API call in async function
    const body = JSON.stringify({ text });

    fetch("http://192.168.0.176:5000/", {
        body
    }).then((res) => console.log(res))
}

export default useGetPredictions;

