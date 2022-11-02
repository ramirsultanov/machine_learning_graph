#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/graph/graphviz.hpp>
#include <opencv2/opencv.hpp>

template <typename Pt>
double distance(const Pt& lhs, const Pt& rhs) {
  const auto xdiff = lhs[0] - rhs[0];
  const auto ydiff = lhs[1] - rhs[1];
  return std::sqrt(xdiff * xdiff + ydiff * ydiff);
}

struct point_t {
  typedef boost::vertex_property_tag kind;
};

typedef std::array<int, 2> Pt;

constexpr Pt bounds = {800, 600};

typedef boost::property<point_t, std::tuple<Pt, int>> PointProperty;
typedef boost::property<boost::edge_weight_t, double> EdgeProperty;
typedef boost::adjacency_list<
    boost::vecS,
    boost::vecS,
    boost::undirectedS,
    PointProperty,
    EdgeProperty
> Graph;
typedef Graph::edge_descriptor Edge;
typedef Graph::vertex_descriptor Vertex;

template <typename Graph, typename Property>
auto find(Graph& g, const Property& pr) {
  typename boost::property_map<Graph, point_t>::type map =
      boost::get(point_t(), g);
  for (auto [it, end] = boost::vertices(g); it != end; it++) {
    if (std::get<Pt>(map[*it]) == pr) {
      return *it;
    }
  }
  return boost::add_vertex(std::make_tuple(pr, 0), g);
}

std::vector<std::vector<Pt>> func(const std::set<Pt>& set, unsigned k) {
  cv::Mat res(bounds[1], bounds[0], CV_8U);
  const auto color = cv::Scalar(255, 255, 255);
  const auto thick = 10;
  std::vector<Pt> data = {set.begin(), set.end()};
  Graph g;
  double min = std::numeric_limits<double>::max();
  std::array<std::size_t, 2> pts;
  for (auto i = 0u; i < data.size(); i++) {
    for (auto j = 0u; j < data.size(); j++) {
      if (i != j) {
        const auto dist = distance(data[i], data[j]);
        if (dist < min) {
          min = dist;
          pts[0] = i;
          pts[1] = j;
        }
      }
    }
  }
  auto uPr = data[pts[0]];
  auto vPr = data[pts[1]];

  const auto u = boost::add_vertex(std::make_tuple(uPr, 0), g);
  const auto v = boost::add_vertex(std::make_tuple(vPr, 0), g);
  boost::add_edge(u, v, min, g);

  for (const auto& d : set) {
    find(g, d);
  }

  std::set<Pt> linked = {data[pts[0]], data[pts[1]]};
  std::set<Pt> isolated;
  for (auto i = 0u; i < data.size(); i++) {
    if (i == pts[0] || i == pts[1]) {
      continue;
    }
    isolated.insert(data[i]);
  }
  while (!isolated.empty()) {
    double min = std::numeric_limits<double>::max();
    std::array<Pt, 2> pts;
    for (
        auto ptl = linked.begin();
        ptl != linked.end();
        ptl++
    ) {
      for (auto pti = isolated.begin(); pti != isolated.end(); pti++) {
        const auto dist = distance(*ptl, *pti);
        if (dist < min) {
          min = dist;
          pts[0] = *ptl;
          pts[1] = *pti;
        }
      }
    }
    const auto u = find(g, pts[0]);
    const auto v = find(g, pts[1]);
    boost::add_edge(u, v, min, g);
    linked.insert(pts[0]);
    linked.insert(pts[1]);
    isolated.erase(pts[0]);
    isolated.erase(pts[1]);
    {
      boost::property_map<Graph, point_t>::type vertexClassMap =
          boost::get(point_t(), g);
      for (auto [it, end] = boost::edges(g); it != end; it++) {
        const auto& s = std::get<Pt>(vertexClassMap[boost::source(*it, g)]);
        const auto& t = std::get<Pt>(vertexClassMap[boost::target(*it, g)]);
        cv::line(res, {s[0], s[1]}, {t[0], t[1]}, color, thick);
      }
      for (auto [it, end] = boost::vertices(g); it != end; it++) {
        const auto& pt = std::get<Pt>(vertexClassMap[*it]);
        cv::circle(res, {pt[0], pt[1]}, thick, color, cv::FILLED);
      }
      cv::imshow("iter", res);
      cv::waitKey(0);
    }
  }
  boost::property_map<Graph, boost::edge_weight_t>::type edgeWeightMap =
      boost::get(boost::edge_weight, g);
  std::size_t clusters = 0;
  {
    auto compare = [&edgeWeightMap](const auto& lhs, const auto& rhs) {
      return edgeWeightMap(lhs) > edgeWeightMap(rhs);
    };
    std::multiset<Edge, decltype(compare)> es(compare);
    boost::graph_traits< Graph >::edge_iterator it, end;
    for (std::tie(it, end) = boost::edges(g); it != end; ++it)
    {
      es.insert(*it);
    }
    boost::property_map<Graph, point_t>::type vertexClassMap =
        boost::get(point_t(), g);
    {
      auto it = es.begin();
      auto end = es.end();
      for (auto i = 0u; i < k; i++) {
        boost::remove_edge(*it, g);
        it++;
      }
      std::cout << std::endl;
      for (auto [it, end] = boost::vertices(g); it != end; it++) {
        const auto adj = boost::adjacent_vertices(*it, g);
        auto cluster = 0;
        for (
            auto [itv, endv] = boost::adjacent_vertices(*it, g);
            itv != endv;
            itv++
        ) {
          const auto& num = std::get<int>(vertexClassMap[*itv]);
          if (num != 0) {
            cluster = num;
            break;
          }
        }
        if (cluster == 0) {
          clusters++;
          cluster = clusters;
        }
        std::get<int>(vertexClassMap[*it]) = cluster;
      }
    }
  }
  res = cv::Mat(bounds[1], bounds[0], CV_8U);
  boost::property_map<Graph, point_t>::type vertexClassMap =
      boost::get(point_t(), g);
  for (auto [it, end] = boost::edges(g); it != end; it++) {
    const auto& s = std::get<Pt>(vertexClassMap[boost::source(*it, g)]);
    const auto& t = std::get<Pt>(vertexClassMap[boost::target(*it, g)]);
    cv::line(res, {s[0], s[1]}, {t[0], t[1]}, color, thick);
  }
  for (auto [it, end] = boost::vertices(g); it != end; it++) {
    const auto& pt = std::get<Pt>(vertexClassMap[*it]);
    cv::circle(res, {pt[0], pt[1]}, thick, color, cv::FILLED);
  }
  cv::destroyWindow("iter");
  cv::imshow("res", res);
  cv::waitKey(0);
  std::vector<std::vector<Pt>> ret(clusters);
  for (auto [it, end] = boost::vertices(g); it != end; it++) {
    const auto pr = vertexClassMap[*it];
    const auto pt = std::get<Pt>(pr);
    const auto num = std::get<int>(pr);
    ret[num - 1].push_back(pt);
  }
  return ret;
}

std::set<Pt> generate(std::size_t n) {
  std::set<Pt> ret;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> disX(0, bounds[0]);
  std::uniform_int_distribution<int> disY(0, bounds[1]);
  while (ret.size() != n) {
    ret.insert({disX(gen), disY(gen)});
  }
  return ret;
}

int main() {
  const auto data = generate(100);
  const auto res = func(data, 3);

  return 0;
}
