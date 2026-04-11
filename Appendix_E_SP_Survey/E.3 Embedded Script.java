Qualtrics.SurveyEngine.addOnPageSubmit(function() {
    console.log("JS started on submit");
    
    var postal = jQuery("#QID18 input").val();
    console.log("postal:", postal);
    
    if(!postal || postal.length < 2){
        console.log("postal invalid, stopping");
        return;
    }
    
    var prefix = postal.substring(0, 2);

    /* postal → planning area */
    var postal_to_planning_area = {
        "01":"DOWNTOWN CORE","02":"DOWNTOWN CORE","03":"OUTRAM","04":"BUKIT MERAH",
        "05":"BUKIT MERAH","06":"DOWNTOWN CORE","07":"OUTRAM","08":"OUTRAM",
        "09":"BUKIT MERAH","10":"BUKIT MERAH","11":"QUEENSTOWN","12":"QUEENSTOWN",
        "13":"CLEMENTI","14":"BUKIT MERAH","15":"QUEENSTOWN","16":"QUEENSTOWN",
        "17":"DOWNTOWN CORE","18":"ROCHOR","19":"ROCHOR","20":"KALLANG",
        "21":"KALLANG","22":"RIVER VALLEY","23":"ORCHARD","24":"TANGLIN",
        "25":"TANGLIN","26":"BUKIT TIMAH","27":"BUKIT TIMAH","28":"NEWTON",
        "29":"NOVENA","30":"NOVENA","31":"TOA PAYOH","32":"TOA PAYOH",
        "33":"SERANGOON","34":"GEYLANG","35":"GEYLANG","36":"GEYLANG",
        "37":"TOA PAYOH","38":"GEYLANG","39":"GEYLANG","40":"GEYLANG",
        "41":"GEYLANG","42":"MARINE PARADE","43":"MARINE PARADE","44":"MARINE PARADE",
        "45":"MARINE PARADE","46":"BEDOK","47":"BEDOK","48":"BEDOK",
        "49":"CHANGI","50":"CHANGI","51":"TAMPINES","52":"TAMPINES",
        "53":"HOUGANG","54":"HOUGANG","55":"SENGKANG","56":"ANG MO KIO",
        "57":"BISHAN","58":"BUKIT TIMAH","59":"BUKIT TIMAH","60":"JURONG WEST",
        "61":"JURONG WEST","62":"JURONG WEST","63":"JURONG WEST","64":"TUAS",
        "65":"BUKIT BATOK","66":"BUKIT PANJANG","67":"CHOA CHU KANG","68":"CHOA CHU KANG",
        "69":"TENGAH","70":"TENGAH","71":"TENGAH","72":"WOODLANDS",
        "73":"WOODLANDS","75":"YISHUN","76":"SEMBAWANG","77":"MANDAI",
        "78":"MANDAI","79":"SELETAR","80":"SELETAR","81":"CHANGI",
        "82":"SENGKANG"
    };

    /* accessibility data */
    var planning_area_accessibility = {
        "ANG MO KIO": {
            "car_time": 15.63333333,
            "car_cost": 3.692378667,
            "pt_time": 36.05,
            "pt_transfers": 1.0,
            "pt_fare": 1.66
        },
        "BEDOK": {
            "car_time": 16.51666667,
            "car_cost": 3.764317333,
            "pt_time": 35.45,
            "pt_transfers": 0.0,
            "pt_fare": 1.66
        },
        "BISHAN": {
            "car_time": 13.98333333,
            "car_cost": 3.554627333,
            "pt_time": 32.8,
            "pt_transfers": 0.0,
            "pt_fare": 1.59
        },
        "BOON LAY": {
            "car_time": 23.41666667,
            "car_cost": 4.309921667,
            "pt_time": 60.91666667,
            "pt_transfers": 1.0,
            "pt_fare": 2.17
        },
        "BUKIT BATOK": {
            "car_time": 20.56666667,
            "car_cost": 4.091440667,
            "pt_time": 44.78333333,
            "pt_transfers": 1.0,
            "pt_fare": 1.85
        },
        "BUKIT MERAH": {
            "car_time": 11.05,
            "car_cost": 3.305998,
            "pt_time": 27.31666667,
            "pt_transfers": 0.0,
            "pt_fare": 1.4
        },
        "BUKIT PANJANG": {
            "car_time": 20.61666667,
            "car_cost": 4.095273667,
            "pt_time": 48.08,
            "pt_transfers": 1.0,
            "pt_fare": 1.81
        },
        "BUKIT TIMAH": {
            "car_time": 15.76666667,
            "car_cost": 3.703237333,
            "pt_time": 38.78333333,
            "pt_transfers": 1.0,
            "pt_fare": 1.59
        },
        "CENTRAL WATER CATCHMENT": {
            "car_time": 28.42666667,
            "car_cost": 4.668768,
            "pt_time": null,
            "pt_transfers": null,
            "pt_fare": null
        },
        "CHANGI": {
            "car_time": 28.08333333,
            "car_cost": 4.644975,
            "pt_time": 56.66,
            "pt_transfers": 1.0,
            "pt_fare": 2.02
        },
        "CHANGI BAY": {
            "car_time": null,
            "car_cost": null,
            "pt_time": null,
            "pt_transfers": null,
            "pt_fare": null
        },
        "CHOA CHU KANG": {
            "car_time": 24.68333333,
            "car_cost": 4.407024333,
            "pt_time": 48.48,
            "pt_transfers": 1.0,
            "pt_fare": 2.07
        },
        "CLEMENTI": {
            "car_time": 16.93333333,
            "car_cost": 3.798250667,
            "pt_time": 41.37333333,
            "pt_transfers": 1.0,
            "pt_fare": 1.86
        },
        "DOWNTOWN CORE": {
            "car_time": 7.1,
            "car_cost": 2.931234,
            "pt_time": 20.26666667,
            "pt_transfers": 0.0,
            "pt_fare": 1.28
        },
        "GEYLANG": {
            "car_time": 11.7,
            "car_cost": 3.361092,
            "pt_time": 27.17333333,
            "pt_transfers": 0.0,
            "pt_fare": 1.59
        },
        "HOUGANG": {
            "car_time": 16.09666667,
            "car_cost": 3.730112533,
            "pt_time": 37.04666667,
            "pt_transfers": 1.0,
            "pt_fare": 1.75
        },
        "JURONG EAST": {
            "car_time": 19.58333333,
            "car_cost": 4.014066667,
            "pt_time": 45.58333333,
            "pt_transfers": 0.0,
            "pt_fare": 1.98
        },
        "JURONG WEST": {
            "car_time": 24.18333333,
            "car_cost": 4.368694333,
            "pt_time": 52.56666667,
            "pt_transfers": 1.0,
            "pt_fare": 2.15
        },
        "KALLANG": {
            "car_time": 10.05,
            "car_cost": 3.221238,
            "pt_time": 23.35,
            "pt_transfers": 0.0,
            "pt_fare": 1.38
        },
        "LIM CHU KANG": {
            "car_time": 33.95,
            "car_cost": 5.019856,
            "pt_time": 32.98333333,
            "pt_transfers": 0.0,
            "pt_fare": 0.0
        },
        "MANDAI": {
            "car_time": 24.93333333,
            "car_cost": 4.426189333,
            "pt_time": 55.61666667,
            "pt_transfers": 1.0,
            "pt_fare": 2.07
        },
        "MARINA EAST": {
            "car_time": 14.17666667,
            "car_cost": 3.571014267,
            "pt_time": 53.85,
            "pt_transfers": 0.0,
            "pt_fare": 0.0
        },
        "MARINA SOUTH": {
            "car_time": 10.6,
            "car_cost": 3.267856,
            "pt_time": 31.06666667,
            "pt_transfers": 0.0,
            "pt_fare": 1.468
        },
        "MARINE PARADE": {
            "car_time": 13.56666667,
            "car_cost": 3.519310667,
            "pt_time": 34.85,
            "pt_transfers": 0.0,
            "pt_fare": 1.68
        },
        "MUSEUM": {
            "car_time": 6.086666667,
            "car_cost": 2.831380133,
            "pt_time": 17.28333333,
            "pt_transfers": 0.0,
            "pt_fare": 1.28
        },
        "NEWTON": {
            "car_time": 8.66,
            "car_cost": 3.0849564,
            "pt_time": 26.35666667,
            "pt_transfers": 0.0,
            "pt_fare": 1.28
        },
        "NORTH-EASTERN ISLANDS": {
            "car_time": null,
            "car_cost": null,
            "pt_time": null,
            "pt_transfers": null,
            "pt_fare": null
        },
        "NOVENA": {
            "car_time": 11.41666667,
            "car_cost": 3.337076667,
            "pt_time": 32.73333333,
            "pt_transfers": 0.0,
            "pt_fare": 1.49
        },
        "ORCHARD": {
            "car_time": 7.736666667,
            "car_cost": 2.993971133,
            "pt_time": 21.32,
            "pt_transfers": 0.0,
            "pt_fare": 1.38
        },
        "OUTRAM": {
            "car_time": 6.95,
            "car_cost": 2.916453,
            "pt_time": 21.44333333,
            "pt_transfers": 0.0,
            "pt_fare": 1.28
        },
        "PASIR RIS": {
            "car_time": 21.44,
            "car_cost": 4.1583904,
            "pt_time": 49.61666667,
            "pt_transfers": 1.0,
            "pt_fare": 1.98
        },
        "PAYA LEBAR": {
            "car_time": 17.16666667,
            "car_cost": 3.817253333,
            "pt_time": 44.45,
            "pt_transfers": 1.0,
            "pt_fare": 1.82
        },
        "PIONEER": {
            "car_time": 27.18333333,
            "car_cost": 4.582605,
            "pt_time": 61.15,
            "pt_transfers": 1.0,
            "pt_fare": 2.33
        },
        "PUNGGOL": {
            "car_time": 20.76666667,
            "car_cost": 4.106772667,
            "pt_time": 44.68333333,
            "pt_transfers": 1.0,
            "pt_fare": 1.94
        },
        "QUEENSTOWN": {
            "car_time": 14.65,
            "car_cost": 3.611134,
            "pt_time": 34.6,
            "pt_transfers": 0.0,
            "pt_fare": 1.68
        },
        "RIVER VALLEY": {
            "car_time": 7.716666667,
            "car_cost": 2.992000333,
            "pt_time": 24.65666667,
            "pt_transfers": 0.0,
            "pt_fare": 1.28
        },
        "ROCHOR": {
            "car_time": 7.5,
            "car_cost": 2.97065,
            "pt_time": 21.41333334,
            "pt_transfers": 0.0,
            "pt_fare": 1.28
        },
        "SELETAR": {
            "car_time": 20.95,
            "car_cost": 4.120827,
            "pt_time": 59.06333334,
            "pt_transfers": 1.0,
            "pt_fare": 1.9
        },
        "SEMBAWANG": {
            "car_time": 28.73333333,
            "car_cost": 4.69002,
            "pt_time": 53.56666667,
            "pt_transfers": 1.0,
            "pt_fare": 2.15
        },
        "SENGKANG": {
            "car_time": 17.43333333,
            "car_cost": 3.838970667,
            "pt_time": 40.66666667,
            "pt_transfers": 1.0,
            "pt_fare": 1.86
        },
        "SERANGOON": {
            "car_time": 14.28333333,
            "car_cost": 3.580055333,
            "pt_time": 34.59666667,
            "pt_transfers": 1.0,
            "pt_fare": 1.68
        },
        "SIMPANG": {
            "car_time": 23.86333333,
            "car_cost": 4.344163133,
            "pt_time": 57.09666667,
            "pt_transfers": 1.0,
            "pt_fare": 2.07
        },
        "SINGAPORE RIVER": {
            "car_time": 6.393333333,
            "car_cost": 2.861599067,
            "pt_time": 21.76666667,
            "pt_transfers": 0.0,
            "pt_fare": 0.0
        },
        "SOUTHERN ISLANDS": {
            "car_time": 22.48333333,
            "car_cost": 4.238372333,
            "pt_time": 65.83666667,
            "pt_transfers": 1.0,
            "pt_fare": 1.82
        },
        "STRAITS VIEW": {
            "car_time": 10.68666667,
            "car_cost": 3.275201867,
            "pt_time": 28.26666667,
            "pt_transfers": 0.0,
            "pt_fare": 1.49
        },
        "SUNGEI KADUT": {
            "car_time": 25.42666667,
            "car_cost": 4.460868,
            "pt_time": 55.19333333,
            "pt_transfers": 1.0,
            "pt_fare": 2.2
        },
        "TAMPINES": {
            "car_time": 19.38333333,
            "car_cost": 3.997778667,
            "pt_time": 41.51666667,
            "pt_transfers": 1.0,
            "pt_fare": 1.94
        },
        "TANGLIN": {
            "car_time": 11.26666667,
            "car_cost": 3.324362667,
            "pt_time": 36.31,
            "pt_transfers": 0.0,
            "pt_fare": 1.49
        },
        "TENGAH": {
            "car_time": 23.5,
            "car_cost": 4.31631,
            "pt_time": 53.83333333,
            "pt_transfers": 1.0,
            "pt_fare": 2.07
        },
        "TOA PAYOH": {
            "car_time": 12.9,
            "car_cost": 3.462804,
            "pt_time": 30.8,
            "pt_transfers": 1.0,
            "pt_fare": 1.59
        },
        "TUAS": {
            "car_time": 32.06,
            "car_cost": 4.9040368,
            "pt_time": 67.95,
            "pt_transfers": 1.0,
            "pt_fare": 2.42
        },
        "WESTERN ISLANDS": {
            "car_time": 44.76,
            "car_cost": 5.7303656,
            "pt_time": null,
            "pt_transfers": null,
            "pt_fare": null
        },
        "WESTERN WATER CATCHMENT": {
            "car_time": 26.44,
            "car_cost": 4.531092,
            "pt_time": 63.66666667,
            "pt_transfers": 1.0,
            "pt_fare": 2.27
        },
        "WOODLANDS": {
            "car_time": 25.31666667,
            "car_cost": 4.453245,
            "pt_time": 51.60666667,
            "pt_transfers": 1.0,
            "pt_fare": 2.15
        },
        "YISHUN": {
            "car_time": 22.73333333,
            "car_cost": 4.257537333,
            "pt_time": 48.95,
            "pt_transfers": 1.0,
            "pt_fare": 1.98
        }
    };

    var area = postal_to_planning_area[prefix];
    var acc = planning_area_accessibility[area];

    if(!acc){ return; }

    /* Format Data */
    var car_time = Math.round(acc.car_time);
    var car_cost = acc.car_cost.toFixed(2);
    var pt_time = Math.round(acc.pt_time);
    var pt_cost = acc.pt_fare.toFixed(2);
    var pt_transfer = 1;

    /* Write into Embedded Data */
    Qualtrics.SurveyEngine.setEmbeddedData("planning_area", area);
    Qualtrics.SurveyEngine.setEmbeddedData("car_time", car_time);
    Qualtrics.SurveyEngine.setEmbeddedData("car_cost", car_cost);
    Qualtrics.SurveyEngine.setEmbeddedData("pt_time", pt_time);
    Qualtrics.SurveyEngine.setEmbeddedData("pt_cost", pt_cost);
    Qualtrics.SurveyEngine.setEmbeddedData("pt_transfer", pt_transfer);

});
