

# Plotly Express使用
#%%
import plotly_express as px
print(px.data.iris.__doc__)
iris = px.data.iris()
#%%
tips = px.data.tips()
gapminder = px.data.gapminder()
election = px.data.election()
wind = px.data.wind()
carshare = px.data.carshare()
px.scatter(iris, x="sepal_width", y="sepal_length")
#%%
px.scatter(iris, x="sepal_width", y="sepal_length", color="species")
#%%
px.scatter(iris, x="sepal_width", y="sepal_length", color="species", marginal_y="violin", marginal_x="rug")

#%%
px.scatter(gapminder.query("year==2007"), x="gdpPercap", y="lifeExp", size="pop", color="continent", hover_name="country", log_x=True, size_max=60)