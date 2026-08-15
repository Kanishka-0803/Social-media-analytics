import sys
sys.path.append("code/")
from social import *

def test_parse_label():
    print("Testing parse_label()...", end="")
    assert(parse_label("From: Steny Hoyer (Representative from Maryland)") == { "name" : "Steny Hoyer", "position" : "Representative", "state" : "Maryland" })
    assert(parse_label("From: Mitch (Senator from Kentucky)") == { "name" : "Mitch", "position" : "Senator", "state" : "Kentucky" })
    assert(parse_label("From: Heidi Heitkamp (Senator from North Dakota)") == { "name" : "Heidi Heitkamp", "position" : "Senator", "state" : "North Dakota" })
    assert(parse_label("From: Chris Collins (Representative from New York)") == { "name" : "Chris Collins", "position" : "Representative", "state" : "New York" })
    assert(parse_label("From: Kelly (Professor from PA)") == { "name" : "Kelly", "position" : "Professor", "state" : "PA" })
    print("... done!")

def test_get_region_from_state():
    print("Testing get_region_from_state()...", end="")
    state_df = pd.read_csv("code/data/statemappings.csv")
    assert(str(get_region_from_state(state_df, "California")) == "West")
    assert(str(get_region_from_state(state_df, "Maine")) == "Northeast")
    assert(str(get_region_from_state(state_df, "Nebraska")) == "Midwest")
    assert(str(get_region_from_state(state_df, "Texas")) == "South")
    print("... done!")

def test_find_hashtags():
    print("Testing find_hashtags()...", end="")
    assert(find_hashtags("I am so #excited to watch #TheMandalorian! #starwars") == [ "#excited", "#TheMandalorian", "#starwars" ])
    assert(find_hashtags("#CMUCarnival will be amazing as long as it doesn't rain #weatherchannel") == [ "#CMUCarnival", "#weatherchannel" ])
    assert(find_hashtags("#Whatif, #everything #is: #hashtags?") ==  [ "#Whatif", "#everything", "#is", "#hashtags" ])
    assert(find_hashtags("I don't like hashtags, I think they're overused") == [ ])
    assert(find_hashtags("So excited for #registration!Let's go CMU!") == [ "#registration" ])
    assert(find_hashtags("I'm nervous-#registration but I think it should work out") == [ "#registration" ])
    assert(find_hashtags("I'm waitlisted for everything #registration...") == [ "#registration" ])
    assert(find_hashtags("Not sure what to take #110#112") == [ "#110", "#112" ])
    assert(find_hashtags("Uh oh#") == ["#"])
    print("... done!")

def test_find_sentiment():
    print("Testing find_sentiment()...", end="")
    classifier = SentimentIntensityAnalyzer()
    assert(find_sentiment(classifier, "great") == (0.6249, "positive"))
    assert(find_sentiment(classifier, "bad") == (-0.5423, "negative"))
    assert(find_sentiment(classifier, "hello") == (0.0, "neutral"))
    assert(find_sentiment(classifier, "Being a senator means getting votes on what you want and also having to take tough votes on what you don't") == (-0.0516, "neutral"))
    assert(find_sentiment(classifier, "Had opportunity to participate in panel on importance of disaster mitigation") == (0.0516, "neutral"))
    assert(find_sentiment(classifier, "If you're in the area tomorrow make sure to stop by my office! I will be hold office hours from 3-4pm CST.") == (0.1007, "positive"))
    assert(find_sentiment(classifier, "The House will pass a bill to pay federal workers for their time in furlough once the shutdown ends.") == (-0.1027, "negative"))
    print("...done!")

def test_add_columns():
    print("Testing add_columns()...", end="")
    df = pd.read_csv("code/data/politicaldata.csv")
    state_df = pd.read_csv("code/data/statemappings.csv")
    add_columns(df, state_df)
    assert(df["name"][1] == "Mitch McConnell")
    assert(df["name"][4] == "Mark Udall")
    assert(df["name"][4979] == "Ted Yoho")
    assert(df["position"][1] == "Senator")
    assert(df["position"][4] == "Senator")
    assert(df["position"][4979] == "Representative")
    assert(df["state"][1] == "Kentucky")
    assert(df["state"][4] == "Colorado")
    assert(df["state"][4979] == "Florida")
    assert(df["region"][1] == "South")
    assert(df["region"][4] == "West")
    assert(df["region"][4979] == "South")
    assert(df["hashtags"][1] == [ "#Obamacare" ])
    assert(df["hashtags"][4] == [ "#drones", "#innovation", "#privacy", "#UAS" ])
    assert(df["hashtags"][4979] == [ ])
    assert(df["sentiment"][0] == "neutral")
    assert(df["sentiment"][1] == "negative")
    assert(df["sentiment"][4978] == "positive")
    assert(df["score"][0] == 0.0)
    assert(df["score"][1] == -0.128)
    assert(df["score"][4978] == 0.3595)
    print("... done!")

def test_get_sentiment_quantiles(df):
    print("Testing get_sentiment_quantiles()...", end="")
    assert(get_sentiment_quantiles(df, "state", "Pennsylvania") == [-0.9196, -0.1779, 0.1779, 0.7424, 0.9793])
    assert(get_sentiment_quantiles(df, "name", "Mitch McConnell") == [-0.5994, -0.2484, 0.0, 0.35045, 0.9042])
    assert(get_sentiment_quantiles(df, "", "") == [-0.9852, 0.0, 0.3678, 0.7003, 0.9981])
    print("... done!")

def test_get_hashtag_subset(df):
    print("Testing get_hashtag_subset()...", end="")
    assert(get_hashtag_subset(df, "state", "Pennsylvania") == {
    '#SOTU', '#ABetterWay', '#PTA', '#WeAre', '#September11',
    '#Forestry', '#military', '#IRSscandal', '#Youth',
    '#whitenosesyndrome', '#Constitution', '#engineering',
    '#cropinsurance', '#Obamacare', '#NationalAdoptionDay', '#4Jobs',
    '#FF', '#EndTrafficking', '#floodinsurance', '#NoDealNoBreak',
    '#OAM2014chat', '#FathersDay', '#spellingbee', '#StopTheSequester',
    '#9', '#EarthDay', '#RateShock', '#MarcellusFest',
    '#corporatewelfare', '#Hezbollah', '#Agriculture',
    '#MarchOnWashington', '#RenewUI-', '#Ebola', '#Sellersville',
    '#manufacturing', '#shalegas', '#tcot', '#BorderCrisis', '#Benghazi',
    '#RestoreTrust', '#Iran', '#NIH', '#CombatSuicide', '#LaborDay',
    '#OpEd', '#House', '#DontDoubleMyRate', '#MedicareAdvantage',
    '#MOW50', '#30Days30Ways', '#MemorialDay', '#America',
    '#WashingtonMonument', '#TrainWreck', '#Vets', '#School', '#Delco',
    '#MentalHealth', '#earmark', '#weather-related', '#FTW', '#THON14',
    '#CBWest', '#BCTHS', '#Safety', '#business', '#humantrafficking',
    '#budget', '#MLKï¿½Ûªs', '#ErieCounty', '#ACA', '#ISIS',
    '#SenateMustAct', '#FortHood', '#SaveSarah', '#HR3717', '#FTK',
    '#PA', '#teens', '#USFS', '#SWPA', '#CoffeeWithKeith',
    '#CareerOneStop', '#IRS', '#DaNicaShirey', '#2013GC', '#energy',
    '#Waterford', "#PA8's", '#jobs', '#Traffic', '#LetsTalk',
    '#ACARepeal', '#USDA' })
    assert(get_hashtag_subset(df, "name", "Mitch McConnell") == {
    '#budget', '#Obamacare', '#ISIL', '#Senate', '#Sequester',
    '#StudentLoans', '#Kentucky', '#SavingCoalJobsAct' })
    assert(len(get_hashtag_subset(df, "region", "West")) == 470) # too long to check all the hashtags here - just check the length instead
    print("... done!")

def test_get_hashtag_rates(df):
    print("Testing get_hashtag_rates()...", end="")
    d = get_hashtag_rates(df)
    assert(len(d) == 1529)
    assert(d["#TrainWreck"] == 8)
    assert(d["#jobs"] == 20)
    assert(d["#STEM"] == 5)
    assert(d["#ObamaCare"] == 20)
    print("... done!")

def test_most_common_hashtags(df):
    print("Testing most_common_hashtags()...", end="")
    d1 = { "#CMU" : 10, "#TheMandalorian" : 15, "#tgif" : 3, "#homework" : 20, "#hashtag" : 1, "#programming" : 7, "#testcase" : 1, "#WorldPeace" : 9, "#coffee" : 18, "#naptime" : 2 }
    assert(most_common_hashtags(d1, 1) == { "#homework" : 20 })
    assert(most_common_hashtags(d1, 2) == { "#homework" : 20, "#coffee" : 18 })
    assert(most_common_hashtags(d1, 5) == { "#homework" : 20, "#coffee" : 18, "#TheMandalorian" : 15, "#CMU" : 10, "#WorldPeace" : 9 })

    d2 = get_hashtag_rates(df)
    assert(most_common_hashtags(d2, 1) == { "#Obamacare" : 61 })
    assert(most_common_hashtags(d2, 7) == { "#Obamacare" : 61, "#IRS" : 26, "#RenewUI" : 21, "#jobs" : 20, "#Benghazi" : 20, "#ObamaCare" : 20, "#SOTU" : 20 })
    print("... done!")

def test_get_hashtag_sentiment(df):
    # Note - we're comparing floats here, so we'll check if they're
    # almost equal instead of exactly equal
    print("Testing get_hashtag_sentiment()...", end="")
    import math
    assert(math.isclose(get_hashtag_sentiment(df, "#TrainWreck"), -0.125))
    assert(math.isclose(get_hashtag_sentiment(df, "#jobs"), 0.7894736842105263))
    assert(math.isclose(get_hashtag_sentiment(df, "#STEM"), 0.6))
    assert(math.isclose(get_hashtag_sentiment(df, "#ObamaCare"), 0))
    print("... done!")

def test_all():
    test_parse_label()
    test_get_region_from_state()
    test_find_hashtags()
    test_find_sentiment()
    test_add_columns()

    df = pd.read_csv("code/data/politicaldata.csv")
    add_columns(df, pd.read_csv("code/data/statemappings.csv"))

    test_get_sentiment_quantiles(df)
    test_get_hashtag_subset(df)
    test_get_hashtag_rates(df)
    test_most_common_hashtags(df)
    test_get_hashtag_sentiment(df)

def run():
    print("\n-----\n")
    print("Now let's look at the general trends!")
    print("\n-----\n")

    df = pd.read_csv("code/data/politicaldata.csv")
    state_df = pd.read_csv("code/data/statemappings.csv")
    add_columns(df, state_df)

    print("Overall Sentiment Quantiles:")
    print(get_sentiment_quantiles(df, "", ""))
    print()

    hashtags = get_hashtag_rates(df)
    print("Total # Hashtags:")
    print(len(get_hashtag_rates(df)))
    print()

    freq_hashtags = most_common_hashtags(hashtags, 10)
    print("Top 10 Hashtags:")
    for hashtag in freq_hashtags:
        print(hashtag, "[", hashtags[hashtag], "uses, average score:", get_hashtag_sentiment(df, hashtag), "]")