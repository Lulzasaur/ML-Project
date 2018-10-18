
const $epochs = $('#epochs')
const $ticker = $('#ticker')
const $newForm = $("#new-form");
const model = tf.sequential()
let epochs,
    ticker;

model.add(tf.layers.conv1d({
  inputShape: [2000, 10],
  kernelSize: 100,
  filters: 8,
  strides: 2,
  activation: 'relu',
  kernelInitializer: 'VarianceScaling'
}))

model.add(tf.layers.conv1d({
  inputShape: [2000, 10],
  kernelSize: 100,
  filters: 8,
  strides: 2,
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

async function train(epo){
  for(let i=0;i<epo;i++){
    const response = await model.fit(xs,ys)
    console.log(response.history.loss[0])
  }
}

$newForm.on("submit", function(e) {
  e.preventDefault();
  
  ticker = $ticker.val()
  epochs = $epochs.val()


  train(epochs).then(()=>{
    console.log('training complete');
    let outputs = model.predict(future_dates);
    outputs.print();
  })

})


