var hollstein_S2 = function(options) {
    // this is script is taken from  https://github.com/fitoprincipe/geetools-code-editor/cloud_masks.js , under MIT licence.
    // * Author: Rodrigo E. Principe
    // * email: fitoprincipe82@gmail.com
    // * Licence: MIT
  // Taken from André Hollstein et al. 2016 (doi:10.3390/rs8080666)
  // http://www.mdpi.com/2072-4292/8/8/666/pdf
  
  var opt = options || ['cloud', 'snow', 'shadow', 'water', 'cirrus']
  
  var difference = function(a, b) {
    var wrap = function(img) {
      return img.select(a).subtract(img.select(b))
    }
    return wrap
  }
  var ratio = function(a, b) {
    var wrap = function(img) {
      return img.select(a).divide(img.select(b))
    }
    return wrap
  }
  
  var opt_list = ee.List(opt)
  
  var compute_dt = function(img) {
    
    //1
    var b3 = img.select('B3').lt(3190)
    
    //2
    var b8a = img.select('B8A').lt(1660)
    var r511 = ratio('B5', 'B11')(img).lt(4.33)
    
    //3
    var s1110 = difference('B11', 'B10')(img).lt(2550)
    var b3_3 = img.select('B3').lt(5250)
    var r210 = ratio('B2','B10')(img).lt(14.689)
    var s37 = difference('B3', 'B7')(img).lt(270)
    
    //4
    var r15 = ratio('B1', 'B5')(img).lt(1.184)
    var s67 = difference('B6', 'B7')(img).lt(-160)
    var b1 = img.select('B1').lt(3000)
    var r29 =  ratio('B2', 'B9')(img).lt(0.788)
    var s911 = difference('B9', 'B11')(img).lt(210)
    var s911_2 = difference('B9', 'B11')(img).lt(-970)
    
    var dtf = dt.binary({1:b3, 
                         21:b8a, 22:r511, 
                         31:s37, 32:r210, 33:s1110, 34:b3_3,
                         41: s911_2, 42:s911, 43:r29, 44:s67, 45:b1, 46:r15
                         },
                         {'shadow-1':[[1,1], [21,1], [31,1], [41,0]],
                          'water':   [[1,1], [21,1], [31,0], [42,1]],
                          'shadow-2':[[1,1], [21,1], [31,0], [42,0]],
                          'cirrus-2':[[1,1], [21,0], [32,1], [43,0]],
                          'cloud-1': [[1,0], [22,1], [33,1], [44,1]],
                          'cirrus-1':[[1,0], [22,1], [33,1], [44,0]],
                          'cloud-2': [[1,0], [22,1], [33,0], [45,0]],
                          'shadow-3':[[1,0], [22,0], [34,1], [46,0]],
                          'snow':    [[1,0], [22,0], [34,0]],
                         }, 'hollstein')
    
    var results = dtf(img)
    
    var optlist = ee.List(opt)
    
    var finalmask = ee.Image(optlist.iterate(function(option, ini){
      ini = ee.Image(ini)
      var mask = results.select([option])
      return ini.or(mask)
    }, ee.Image.constant(0).select([0], ['hollstein'])))
    
    return img.addBands(results).updateMask(finalmask.not())//results.select('hollstein'))
  }
  return compute_dt
}
