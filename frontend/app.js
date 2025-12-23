const { createApp, ref, nextTick } = Vue;

createApp({
  setup() {
    const loading = ref(false);
    const error = ref('');
    const result = ref(null);
    const chart = ref(null);
    const selectedEquation = ref('eq1');

    // Функции форматирования
    const formatX = (value) => {
      return parseFloat(value).toFixed(2);
    };

    const formatY = (value) => {
      return parseFloat(value).toFixed(4);
    };

    const solve = async () => {
      loading.value = true;
      error.value = '';
      try {
        const res = await fetch('http://127.0.0.1:8080/solve', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ equation: selectedEquation.value })
        });
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        result.value = data;
        await nextTick();
        renderChart(data);
      } catch (e) {
        error.value = 'Ошибка: ' + e.message;
      } finally {
        loading.value = false;
      }
    };

    const renderChart = (data) => {
      const canvas = chart.value;
      if (!canvas) return;
      
      const existingChart = Chart.getChart(canvas);
      if (existingChart) existingChart.destroy();

      // Округляем значения X для подписей
      const formattedLabels = data.x.map(x => parseFloat(x).toFixed(2));

      new Chart(canvas, {
        type: 'line',
        data: {
          labels: formattedLabels,
          datasets: [
            {
              label: 'Численное решение (RK45)',
              data: data.numerical.map(y => parseFloat(y).toFixed(4)),
              borderColor: '#ff6b35',
              backgroundColor: 'rgba(255, 107, 53, 0.1)',
              borderWidth: 3,
              pointRadius: 0,
              pointHoverRadius: 5,
              tension: 0.3,
              fill: false
            },
            {
              label: 'Нейросетевое решение (PINN)',
              data: data.neural.map(y => parseFloat(y).toFixed(4)),
              borderColor: '#00d4ff',
              backgroundColor: 'rgba(0, 212, 255, 0.1)',
              borderWidth: 3,
              borderDash: [5, 5],
              pointRadius: 0,
              pointHoverRadius: 5,
              tension: 0.3,
              fill: false
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              position: 'top',
              labels: {
                color: '#ffffff',
                font: {
                  size: 14,
                  family: "'Segoe UI', sans-serif"
                },
                padding: 20
              }
            },
            tooltip: {
              mode: 'index',
              intersect: false,
              backgroundColor: 'rgba(30, 30, 30, 0.9)',
              titleColor: '#ff6b35',
              bodyColor: '#ffffff',
              borderColor: '#ff6b35',
              borderWidth: 1,
              callbacks: {
                label: function(context) {
                  let label = context.dataset.label || '';
                  if (label) {
                    label += ': ';
                  }
                  label += parseFloat(context.raw).toFixed(4);
                  return label;
                }
              }
            }
          },
          scales: {
            x: {
              grid: {
                color: 'rgba(255, 255, 255, 0.1)',
                borderColor: 'rgba(255, 255, 255, 0.1)'
              },
              ticks: {
                color: '#b0b0b0',
                font: {
                  size: 12
                }
              },
              title: {
                display: true,
                text: 'x',
                color: '#ff6b35',
                font: {
                  size: 16,
                  weight: 'bold'
                }
              }
            },
            y: {
              grid: {
                color: 'rgba(255, 255, 255, 0.1)',
                borderColor: 'rgba(255, 255, 255, 0.1)'
              },
              ticks: {
                color: '#b0b0b0',
                font: {
                  size: 12
                },
                callback: function(value) {
                  return parseFloat(value).toFixed(2);
                }
              },
              title: {
                display: true,
                text: 'y(x)',
                color: '#00d4ff',
                font: {
                  size: 16,
                  weight: 'bold'
                }
              }
            }
          },
          interaction: {
            intersect: false,
            mode: 'nearest'
          },
          animation: {
            duration: 1000,
            easing: 'easeOutQuart'
          }
        }
      });
    };

    return { 
      loading, 
      error, 
      result, 
      chart, 
      solve, 
      selectedEquation,
      formatX,
      formatY
    };
  }
}).mount('#app');