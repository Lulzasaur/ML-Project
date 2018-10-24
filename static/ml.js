
const $epochs = $('#epochs')
const $ticker = $('#ticker')
const $newForm = $("#new-form");
const BASE_URL = 'https://api.iextrading.com/1.0'
const model = tf.sequential()
let epochs,
    ticker;

async function train(epo,xs,ys){
  for(let i=0;i<epo;i++){
    const response = await model.fit(xs,ys)
    console.log(response.history.loss[0])
  }
}

async function getStockInfo(ticker,epochs){
  //get data
  let stockInfo = await $.get(`${BASE_URL}/stock/${ticker}/chart/6m`, function(response) {

    //prep data for machine learning...get array with just date and close
    let dateAndClose = response.map(r => [[Date.parse(r.date)],[r.close]])  

    //if close in next 7 days is greater than current day close by 1%, assign a value of 1
    //if close in next 7 days is less than current day close by 1%, assign a value of -1
    //otherwise assign value of 0, unchanged. 
    for(let i=0;i<dateAndClose.length-7;i++){
      let futureClosePrice = dateAndClose[i+7][1],
          currentClosePrice = dateAndClose[i][1],
          percentClose = (futureClosePrice - currentClosePrice)/currentClosePrice;

      if(percentClose >= 0.01){
        dateAndClose[i].push([1])
      } else if (percentClose <= -0.01){
        dateAndClose[i].push([-1])
      } else {
        dateAndClose[i].push([0])
      }
    }
    //lol gross.
    dateAndClose.pop()
    dateAndClose.pop()
    dateAndClose.pop()
    dateAndClose.pop()
    dateAndClose.pop()
    dateAndClose.pop()
    dateAndClose.pop()

    let tDateAndClose = tf.tensor(dateAndClose)

    model.add(tf.layers.conv1d({
      inputShape: [tDateAndClose.shape],
      kernelSize: 100,
      filters: 8,
      strides: 2,
      activation: 'relu',
      kernelInitializer: 'VarianceScaling'
    }))

    model.add(tf.layers.maxPooling1d({
      poolSize: [500],
      strides: [2]
    }))

    model.add(tf.layers.conv1d({
      kernelSize: 5,
      filters: 16,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'VarianceScaling'
    }))

    model.add(tf.layers.maxPooling1d({
      poolSize: [100],
      strides: [2]
    }))

    model.add(tf.layers.dense({
      units: 10,
      kernelInitializer: 'VarianceScaling',
      activation: 'softmax'
    }))
    
    model.compile({optimizer: 'sgd', loss: 'binaryCrossentropy', lr: 0.1})
    model.fit({batchSize: 3,
      epochs: 10})
    train(epochs,tDateAndClose)
  });

}

$newForm.on("submit", function(e) {
  e.preventDefault();
  
  ticker = $ticker.val()
  epochs = $epochs.val()

  getStockInfo(ticker,epochs)

})


