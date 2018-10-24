
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
    let dateAndClose = response.map(r => [[Date.parse(r.date)],[r.close], [response.length]])
    console.log(dateAndClose)
    let tDateAndClose = tf.tensor3d(dateAndClose)
    
    console.log(tDateAndClose.shape)

    // const lstm = tf.layers.lstm({units: 8, returnSequences: true});

    // // Create an input with 10 time steps.
    // const input = tf.input({shape: [10, 20]});
    // const output = lstm.apply(input);
    
    // console.log(JSON.stringify(output.shape));

    //Juan model
    const lstm = tf.layers.lstm({units: 8, returnSequences: true});
    const input = tf.input({shape: tDateAndClose.shape })
    const output = lstm.apply(input)

    console.log(lstm)
    console.log(input)
    console.log('CHECK THIS OUT', JSON.stringify(output.shape));
    // model.add(tf.layers.lstm({}))


    model.add(lstm)
    
    // model.add(lstm)
  
    model.add(tf.layers.dense({
      units: 2,
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


