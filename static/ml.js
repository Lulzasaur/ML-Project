
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
  let stockInfo = await $.get(`${BASE_URL}/stock/${ticker}/chart/6m`, function(response) {
    let dates = response.map(r => Date.parse(r.date))
    let prices = response.map(r => r.close)
    let tdates = tf.tensor1d(dates)
    let tprices = tf.tensor1d(prices)

    tdates.print()
    tprices.print()

    model.add(tf.layers.conv1d({
      inputShape: [dates.length, 1],
      kernelSize: 10,
      filters: 8,
      strides: 2,
      activation: 'relu',
      kernelInitializer: 'VarianceScaling'
    }))
    
    model.add(tf.layers.conv1d({
      inputShape: [dates.length, 1],
      kernelSize: 10,
      filters: 8,
      strides: 2,
      activation: 'relu',
      kernelInitializer: 'VarianceScaling'
    }))
    
    model.add(tf.layers.maxPooling1d({
      poolSize: [23],
      strides: [2]
    }))

    model.add(tf.layers.dense({
      units: 1,
      kernelInitializer: 'VarianceScaling',
      activation: 'softmax'
    }))
    
    model.compile({optimizer: 'sgd', loss: 'binaryCrossentropy', lr: 0.1})
    
    train(epochs,tdates,tprices)
  });

}

$newForm.on("submit", function(e) {
  e.preventDefault();
  
  ticker = $ticker.val()
  epochs = $epochs.val()

  getStockInfo(ticker,epochs)

})


