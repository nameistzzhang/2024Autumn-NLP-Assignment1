# 2024Autumn-NLP-Assignment1

## Task1

- `flatten_list(nested_list: list)`

Since we cannot determine how many nested levels the input list has, we need to use a recursive function to flatten the list. The function `flatten_list` takes a nested list as input and returns a flattened list.

And here we present some experiment results when processing a larger scale of nested list. **Each column of the table has the same nesting layer number, ranging from 1 to 10. Each row of the table has the same number of elements, ranging from 1e2 to 1e6.**

<table>
  <tr>
    <th style="background-color: #FFFFFF; "></th>
    <th style="background-color: #d6d6fe; ">1</th>
    <th style="background-color: #d6d6fe; ">2</th>
    <th style="background-color: #d6d6fe; ">3</th>
    <th style="background-color: #d6d6fe; ">4</th>
    <th style="background-color: #d6d6fe; ">5</th>
    <th style="background-color: #d6d6fe; ">6</th>
    <th style="background-color: #d6d6fe; ">7</th>
    <th style="background-color: #d6d6fe; ">8</th>
    <th style="background-color: #d6d6fe; ">9</th>
    <th style="background-color: #d6d6fe; ">10</th>
  </tr>
  <tr>
    <th style="background-color: #fe6062;">1e2</td>
    <td style="background-color: #FFFFFF;">0.000</td>
    <td style="background-color: #FFFFFF;">0.000</td>
    <td style="background-color: #FFFFFF;">0.000</td>
    <td style="background-color: #FFFFFF;">0.000</td>
    <th style="background-color: #061c6d; color: #FFFFFF">0.001</td>
    <td style="background-color: #FFFFFF;">0.000</td>
    <td style="background-color: #FFFFFF;">0.000</td>
    <td style="background-color: #FFFFFF;">0.001</td>
    <td style="background-color: #FFFFFF;">0.000</td>
    <td style="background-color: #FFFFFF;">0.000</td>
  </tr>
  <tr>
    <th style="background-color: #fe6062;">1e3</td>
    <td style="background-color: #FFFFFF;">0.000</td>
    <td style="background-color: #FFFFFF;">0.000</td>
    <th style="background-color: #061c6d; color: #FFFFFF">0.004</td>
    <td style="background-color: #FFFFFF;">0.000</td>
    <td style="background-color: #FFFFFF;">0.000</td>
    <td style="background-color: #FFFFFF;">0.003</td>
    <td style="background-color: #FFFFFF;">0.001</td>
    <td style="background-color: #FFFFFF;">0.002</td>
    <td style="background-color: #FFFFFF;">0.001</td>
    <td style="background-color: #FFFFFF;">0.001</td>
  </tr>
  <tr>
    <th style="background-color: #fe6062;">1e4</td>
    <td style="background-color: #FFFFFF;">0.000</td>
    <td style="background-color: #FFFFFF;">0.004</td>
    <td style="background-color: #FFFFFF;">0.004</td>
    <td style="background-color: #FFFFFF;">0.012</td>
    <td style="background-color: #FFFFFF;">0.007</td>
    <td style="background-color: #FFFFFF;">0.017</td>
    <td style="background-color: #FFFFFF;">0.014</td>
    <td style="background-color: #FFFFFF;">0.016</td>
    <td style="background-color: #FFFFFF;">0.016</td>
    <th style="background-color: #061c6d; color: #FFFFFF">0.030</td>
  </tr>
  <tr>
    <th style="background-color: #fe6062;">1e5</td>
    <td style="background-color: #FFFFFF;">0.007</td>
    <td style="background-color: #FFFFFF;">0.032</td>
    <td style="background-color: #FFFFFF;">0.045</td>
    <td style="background-color: #FFFFFF;">0.059</td>
    <td style="background-color: #FFFFFF;">0.078</td>
    <td style="background-color: #FFFFFF;">0.105</td>
    <td style="background-color: #FFFFFF;">0.129</td>
    <th style="background-color: #061c6d; color: #FFFFFF">0.157</td>
    <td style="background-color: #FFFFFF;">0.146</td>
    <td style="background-color: #FFFFFF;">0.157</td>
  </tr>
  <tr>
    <th style="background-color: #fe6062;">1e6</td>
    <th style="background-color: #900C3F; color: #FFFFFF;">0.093</td>
    <th style="background-color: #900C3F; color: #FFFFFF;">0.252</th>
    <th style="background-color: #900C3F; color: #FFFFFF;">0.405</th>
    <th style="background-color: #900C3F; color: #FFFFFF;">0.594</th>
    <th style="background-color: #900C3F; color: #FFFFFF;">0.732</th>
    <th style="background-color: #900C3F; color: #FFFFFF;">0.892</th>
    <th style="background-color: #900C3F; color: #FFFFFF;">1.038</th>
    <th style="background-color: #900C3F; color: #FFFFFF;">1.181</th>
    <th style="background-color: #900C3F; color: #FFFFFF;">1.377</th>
    <th style="background-color: #900C3F; color: #FFFFFF;">1.689</th>
  </tr>
</table>

Obviously, we can see that the time cost of the function **increases** with the number of elements and the nesting layer number. But the increase is **not apparent when the element number or the nesting layer number is small**.「Indicating that the increase is **not linear**.」

- `char_count(s: str)`

We try to use a dictionary to store the frequency of each character in the input string. The function `char_count` takes a string as input and returns a dictionary with the frequency of each character.

<table>
  <tr>
    <th style="background-color: #FFFFFF;">scale</td>
    <th style="background-color: #d6d6fe;">1e1</td>
    <th style="background-color: #d6d6fe;">1e2</td>
    <th style="background-color: #d6d6fe;">1e3</td>
    <th style="background-color: #d6d6fe;">1e4</td>
    <th style="background-color: #d6d6fe;">1e5</td>
    <th style="background-color: #d6d6fe;">1e6</td>
    <th style="background-color: #d6d6fe;">1e7</td>
    <th style="background-color: #d6d6fe;">1e8</td>
  </tr>
  <tr>
    <th style="background-color: #fe6062;">time cost</td>
    <td style="background-color: #FFFFFF;">0.000000</td>
    <td style="background-color: #FFFFFF;">0.000000</td>
    <td style="background-color: #FFFFFF;">0.000000</td>
    <td style="background-color: #FFFFFF;">0.000999</td>
    <td style="background-color: #FFFFFF;">0.029025</td>
    <td style="background-color: #FFFFFF;">0.127361</td>
    <td style="background-color: #FFFFFF;">1.161906</td>
    <th style="background-color: #900C3F; color: #FFFFFF">12.462080</th>
  </tr>
</table>

We can see that the time cost of the function **increases** with the length of the input string. And the increase is **not linear**. See the picture below for better understanding.

![char_count](./rsc/pic/char%20count%20time%20cost.png)
