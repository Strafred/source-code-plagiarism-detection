import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
public class fungsi {
    public static void main(String[] args) {
         System.out.print("Enter an integer: ");
        Scanner sc = new Scanner(System.in);
        int   input;
        input =    sc.nextInt();
        terbalik(input);
    }
    public static void terbalik(int angka)  {
        List arr =new ArrayList();
        for(int i=0;i<=angka;i++) {
             int   sisa ;
             sisa= angka % 10;
             arr.add(sisa);
             angka = angka / 10;
             i=0;
        }
        for(int i=0;i<arr.size();i++)
        {
            System.out.print(arr.get(i));
        }
        System.out.println();
    }
}
