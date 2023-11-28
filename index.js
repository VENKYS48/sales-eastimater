let form = document.querySelector("form")
let result = document.querySelector("#result")


form.addEventListener("submit", (e) => {
    // prevent Page from reloading 
    e.preventDefault()

    const fromData = new FormData(form)

    fetch("http://http://20.84.116.88/predict", {
        method: "POST",
        body: fromData
    }).then((res) => {
        return res.json()
    }).then((data) => { 
        console.log(data)
        result.innerText = data.result
     }, console.error)
})