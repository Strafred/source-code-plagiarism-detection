import java.text.DecimalFormat;
import java.util.Scanner;
public class T02 {
    public static void main(String[] args) {
        Scanner s = new Scanner(System.in);
        System.out.print("Enter the radius and length of a cylinder :");
        double radius = s.nextDouble();
        double length = s.nextDouble();
        double area = radius *radius * 3.14159;
        DecimalFormat df = new DecimalFormat(".####");
        DecimalFormat df2= new DecimalFormat(".#");
        double volume = area * length;
        System.out.println("The area is "+df.format(area));
        System.out.println("The volume is "+df2.format(volume));
    }
}
